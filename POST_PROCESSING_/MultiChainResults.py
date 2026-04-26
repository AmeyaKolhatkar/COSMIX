# MultiChainResults

import numpy as np
from POST_PROCESSING_.Diagnostics import GR_rhat, multichain_tau_ess

class MultiChainResults:
    def __init__(self, chains, log_probs, param_names):
        """
        chains: list of arrays, each (nsamples, ndim)
        log_probs: list of arrays, each (nsamples, )
        """
        self.chains = chains
        self.log_probs = log_probs
        self.param_names = param_names

        self.nchains = len(chains)
        self.ndim = chains[0].shape[1]
        self._rhat = GR_rhat(self.chain_array)
        self._tau, self._ess = multichain_tau_ess(self.chain_array)

    @property
    def stacked_chain(self):
        return np.vstack(self.chains)
    
    @property
    def stacked_log_prob(self):
        return np.concatenate(self.log_probs)
    
    @property
    def chain_array(self):
        """
        Shape: (nchains, nsamples, ndim)
        Required for rhat
        """
        return np.stack(self.chains, axis=0)
    
    @property
    def rhat(self):
        return self._rhat

    @property
    def tau(self):
        return self._tau

    @property
    def ess(self):
        return self._ess
    
    def max_rhat(self, rhat):
        return np.max(rhat)
    
    def is_converged(self, tol=0.01, ess_min=None, tau_factor=None):
        """
        Multi-criteria convergence check.

        Parameters
        ----------
        tol : float
            R-hat tolerance (converged when max(R-hat - 1) < tol).
        ess_min : float or None
            Minimum ESS per parameter. If None, ESS check is skipped.
        tau_factor : float or None
            Require N_per_chain > tau_factor * tau. If None, tau check is skipped.

        Returns
        -------
        bool
        """
        rhat_ok = np.all(self._rhat - 1 < tol)

        if ess_min is not None:
            ess_ok = np.all(self._ess > ess_min)
        else:
            ess_ok = True

        if tau_factor is not None:
            n_per_chain = self.chains[0].shape[0]
            tau_ok = np.all(n_per_chain > tau_factor * self._tau)
        else:
            tau_ok = True

        return rhat_ok and ess_ok and tau_ok

    def convergence_detail(self, tol=0.01, ess_min=None, tau_factor=None):
        """Return a dict with per-criterion pass/fail and values."""
        n_per_chain = self.chains[0].shape[0]
        detail = {
            "rhat_ok": bool(np.all(self._rhat - 1 < tol)),
            "rhat_max": float(np.max(self._rhat)),
        }
        if ess_min is not None:
            detail["ess_ok"] = bool(np.all(self._ess > ess_min))
            detail["ess_min_val"] = float(np.min(self._ess))
        if tau_factor is not None:
            detail["tau_ok"] = bool(np.all(n_per_chain > tau_factor * self._tau))
            detail["tau_max"] = float(np.max(self._tau))
        return detail