"""Diagnostics — convergence and chain quality statistics.

Provides per-parameter diagnostics for a completed chain:

    autocorr()          — integrated autocorrelation time τ per parameter.
    ess()               — effective sample size N / τ.
    is_converged_single(ess_min)
                        — single-chain convergence check (ESS and τ criteria).
    is_converged_multi(rhat_tol)
                        — multi-chain convergence via the Gelman-Rubin R̂ statistic.
    summary()           — prints a table of ESS and τ for all free parameters.

Also exposes two low-level functions used by MultiChainResults:
    _fft_autocorr(x)    — FFT-based normalized autocorrelation.
    _integrated_tau(acf)— Geyer initial positive sequence estimator for τ.
"""
from POST_PROCESSING_.ResultsContainer import MCMCResults
import numpy as np

class MCMCDiagnostics:
    def __init__(self, results: MCMCResults):
        self.results = results

    def autocorr(self):
        tau = self.results.tau
        if tau is None:
            raise RuntimeError("Autocorrelation time no available or not converged")
        return tau

    def ess(self):
        return self.results.ess

    def is_converged_single(self, ess_min=1000):
        ess = self.ess()
        tau = self.autocorr()
        n = self.results.chain.shape[0]

        return ess is not None and np.all(ess > ess_min) and np.all(n > 50*tau)
    
    def is_converged_multi(self, rhat_tol=0.01):
        if not hasattr(self.results, "rhat"):
            return False

        return np.all(self.results.rhat - 1 < rhat_tol)
    
    def summary(self):
        ess = self.ess()
        tau = self.autocorr()
        print("-"*75)
        for i, name in enumerate(self.results.param_names):
            print(
                f"{name:10s}"
                f"ESS: {ess[i]:.4f}"
                f"      Tau: {tau[i]:.4f}"
            )

def _fft_autocorr(x):
    """FFT-based normalized autocorrelation for a 1D array."""
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)          # zero-pad for linear correlation
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0:
        return np.zeros(n)
    return acf / acf[0]


def _integrated_tau(acf):
    """
    Integrated autocorrelation time.
    Uses Geyer's initial positive sequence: sums consecutive pairs of
    autocorrelations and stops when a pair sum goes negative.
    """
    tau = 1.0
    n = len(acf)
    t = 1
    while t < n - 1:
        pair_sum = acf[t] + acf[t + 1]
        if pair_sum < 0:
            break
        tau += 2.0 * pair_sum
        t += 2
    return max(tau, 1.0)


def multichain_tau_ess(chains):
    """
    Compute per-parameter integrated autocorrelation time and effective sample
    size for a multi-chain ensemble.

    Parameters
    ----------
    chains : array-like, shape (nchains, nsamples, ndim)

    Returns
    -------
    tau : ndarray, shape (ndim,)
        Pooled integrated autocorrelation time per parameter.
    ess : ndarray, shape (ndim,)
        Effective sample size per parameter (total across all chains).

    Method
    ------
    1. Per chain, compute the normalized ACF via FFT.
    2. Average ACFs across chains → pooled ACF.
    3. Integrate the pooled ACF (Geyer's initial positive sequence) → τ.
    4. ESS = N_total / τ.
    """
    chains = np.asarray(chains)
    if chains.ndim != 3:
        raise ValueError("chains must have shape (nchains, nsamples, ndim)")

    nchains, nsamples, ndim = chains.shape
    tau = np.zeros(ndim)

    for d in range(ndim):
        acf_pooled = np.zeros(nsamples)
        for j in range(nchains):
            acf_pooled += _fft_autocorr(chains[j, :, d])
        acf_pooled /= nchains
        tau[d] = _integrated_tau(acf_pooled)

    total_samples = nchains * nsamples
    ess = total_samples / tau

    return tau, ess


def GR_rhat(chains):
        """
        Compute Gelman-Rubin R hat statistics.
        """
        if isinstance(chains, (list, tuple)):
            chains = [np.asarray(c) for c in chains]
            for i, c in enumerate(chains):
                if c.ndim != 2:
                    raise ValueError(f"Each chain must be 2D (nsamples, ndim). Chain {i} has shape {c.shape}")
            nsamples_list = [c.shape[0] for c in chains]
            if len(set(nsamples_list)) != 1:
                raise ValueError(f"All chains must have equal length for rhat. Chain lengths: {nsamples_list}")
            chains = np.stack(chains, axis=0)

        chains = np.asarray(chains)
        if chains.ndim != 3:
            raise ValueError("chains must have shape: (nchains, nsamples, ndim)")
        
        nchains, nsamples, ndim = chains.shape
        if nchains < 2:
            raise ValueError("Gelman-Rubin requires at least two chains.")
        
        # split each chain in half
        half = nsamples // 2
        chains = np.concatenate(
            [chains[:, :half, :], chains[:, half:2*half, :]], axis=0
        )

        nchains = chains.shape[0]
        nsamples = chains.shape[1]
        if nsamples < 20:
            raise RuntimeError("Too few samples per chain to compute rhat reliably.")
        
        chains_mean = np.mean(chains, axis=1)               # (nchains, ndim)
        chains_variance = np.var(chains, axis=1, ddof=1)    # (nchains, ndim)

        W = np.mean(chains_variance, axis=0)                # within chain variance
        W = np.maximum(W, 1e-12)
        B = nsamples * np.var(chains_mean, axis=0, ddof=1)  # between chain variance

        V_hat = ( (nsamples - 1) / nsamples ) * W + B / nsamples
        rhat = np.sqrt(V_hat/W)

        return rhat