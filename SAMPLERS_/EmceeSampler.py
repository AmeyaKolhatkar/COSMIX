"""EmceeSampler — emcee ensemble MCMC wrapper.

Uses the affine-invariant ensemble sampler from the `emcee` package.
Ideal for low-to-medium dimensional posteriors (typically < 30 parameters)
where the posterior is unimodal or mildly multimodal.

Typical usage
-------------
sampler = emceeSampler(pm=pipeline.pm, lnpost=pipeline.lnposterior)
result  = sampler.run(nsteps=5000, burn_in=1000)
# result is a dict with keys: chain, log_prob, best_fit, acceptance, tau
"""
import numpy as np
import emcee
from SAMPLERS_.SamplerBase import SamplerBase

class emceeSampler(SamplerBase):
    """
    emcee wrapper.

    responsibilities:
        1. Inherit automated optimization from SamplerBase
        2. initialize walkers at the MAP
        3. run emcee with optional two-phase burn-in
        4. return chains and diagnostics

    Parameters
    ----------
    moves : emcee.moves object or list, optional
        Proposal strategy passed to EnsembleSampler.  Default (None) uses
        emcee's StretchMove.  For correlated posteriors consider:
            moves=[emcee.moves.DEMove(), emcee.moves.DESnookerMove()]
    random_seed : int or None
        Seeds a local numpy Generator — does NOT mutate the global RNG state.
    """

    def __init__(self, pm, lnpost, nwalkers=None, random_seed=None, moves=None, verbose=True, initial_walkers=None, norm_func=None):
        super().__init__(pm, lnpost, verbose=verbose, norm_func=norm_func)

        if nwalkers is None:
            # DEMove needs ndim+1 complementary walkers; 3×ndim gives a
            # comfortable margin while cutting per-step cost vs the old 5×ndim.
            self.nwalkers = max(3*self.ndim, 20)
        else:
            self.nwalkers = nwalkers

        # Local Generator — never touches the global numpy RNG state
        self._rng = np.random.default_rng(random_seed)

        # Default: 80% DEMove + 20% DESnookerMove.
        # StretchMove (emcee default) fails on correlated posteriors — all walkers
        # collapse into the degenerate valley and the stretch scale is wrong.
        # DEMove uses differential vectors between random walker pairs, so it
        # adapts to the actual posterior covariance (including tight correlations
        # like sigma80–lambda0).  DESnookerMove adds out-of-plane exploration.
        if moves is None:
            self._moves = [
                (emcee.moves.DEMove(),        0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ]
        else:
            self._moves = moves

        if initial_walkers is not None:
            # Resume from provided walker positions — skip Pre-Flight Optimizer
            self.p0 = np.asarray(initial_walkers, dtype=float)
            self.nwalkers = self.p0.shape[0]
        else:
            self._initialize_walkers()

    def _safe_lnpost(self, theta):
        val = self.lnpost(theta)
        return val if np.isfinite(val) else -np.inf

    def _initialize_walkers(self):
        best_fit = self.optimize_starting_position()

        # Per-parameter scales instead of one hardcoded constant
        scales = np.array([p.proposed_scale for p in self.pm.free_params])

        p0 = []
        attempts = 0
        max_attempts = 10000

        if self.verbose:
            print("[EmceeSampler] Deploying walkers . . .")
        while len(p0) < self.nwalkers and attempts < max_attempts:
            cand = best_fit + scales * 0.5 * self._rng.standard_normal(self.ndim)

            # hard bounds from Parameter Manager
            valid = True
            for i, p in enumerate(self.pm.free_params):
                if not p.prior.in_support(cand[i]):
                    valid = False
                    break

            if valid:
                lp = self.pm.lnprior(cand)
                if np.isfinite(lp):
                    p0.append(cand)

            attempts += 1

        if len(p0) < self.nwalkers:
            raise RuntimeError(
                f"[EmceeSampler] Failed to initialize {self.nwalkers} walkers after {attempts} attempts"
            )

        self.p0 = np.asarray(p0)

    def run(self, nsteps, burn_in=0, progress=True):
        # Store sampler on self so callers can inspect the full chain later
        self._sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self._safe_lnpost,
            moves=self._moves
        )

        # Phase 1: burn-in — reset afterwards so τ is computed on production chain only
        # progress and prints are gated behind verbose so that only chain 0 produces
        # output when running in parallel — avoids interleaved tqdm bars from subprocesses
        _show = progress and self.verbose
        if burn_in > 0:
            if self.verbose:
                print(f"[EmceeSampler] Running {burn_in} burn-in steps . . .")
            state = self._sampler.run_mcmc(self.p0, burn_in, progress=_show)
            self._sampler.reset()
            if self.verbose:
                print(f"[EmceeSampler] Burn-in complete. Starting production chain . . .")
        else:
            state = self.p0

        # Phase 2: production run
        self._sampler.run_mcmc(state, nsteps, progress=_show)

        chain    = self._sampler.get_chain(flat=True)
        log_prob = self._sampler.get_log_prob(flat=True)

        acceptance = np.mean(self._sampler.acceptance_fraction)

        try:
            tau = self._sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            tau = None
            print("[EmceeSampler] Warning! Chain too short to compute reliable autocorrelation time")

        best_idx = np.argmax(log_prob)
        best_fit = chain[best_idx]

        # Final walker positions — used by MultiChainDriver for continuous runs
        final_state = self._sampler.get_last_sample().coords  # (nwalkers, ndim)

        return {
            "chain": chain,
            "log_prob": log_prob,
            "acceptance": acceptance,
            "best_fit": best_fit,
            "tau": tau,
            "final_state": final_state
        }
