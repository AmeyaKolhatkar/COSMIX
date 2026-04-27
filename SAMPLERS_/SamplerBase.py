"""SamplerBase — shared infrastructure for all COSMIX samplers.

Provides:

    optimize_starting_position()
        Runs a gradient-free scipy optimizer (L-BFGS-B + basin-hopping
        fallback) to find the MAP before a sampler starts.  This ensures
        walkers/live-points begin near the probability peak rather than
        in an empty region, dramatically reducing burn-in.

    _find_valid_spawn(initial_guess, scales, bounds, objective)
        Throws random jitter around the starting guess until a physically
        valid point (finite posterior) is found.  Used when the exact
        initial guess is in an unphysical region.

All concrete samplers (emceeSampler, DynestySampler, PolyChordSampler)
inherit this class and call super().__init__(pm, lnpost) to set up the
shared attributes (pm, lnpost, ndim).
"""
import numpy as np
from scipy.optimize import minimize

class SamplerBase:
    """
    Base class for all MCMC/Nested Sampling Samplers.
    Provides automated parameter extraction and Pre-flight MAP (Maximum A Posteriori) optimization
    capable of handling unbounded priors and fractured parameter spaces.
    """
    def __init__(self, pm, lnpost, verbose=True):
        self.pm = pm
        self.lnpost = lnpost
        self.ndim = len(self.pm.free_params)
        self.verbose = verbose

    def _get_dynamical_guess_and_bounds(self):
        """
        Builds a starting guess, valid bounds, and proposal scales.
        Safely handles infinite/unbounded priors.
        """
        guess = []
        bounds = []
        scales = []

        for p in self.pm.free_params:
            # Handle starting guess (reference point)
            if hasattr(p.prior, "mean") and p.prior.mean is not None:
                guess.append(p.prior.mean)
            elif hasattr(p, "value") and p.value is not None:
                # If unbounded, use the fiducial value defined in the parameter definition
                guess.append(p.value)
            else: 
                # Fallback for bounded uniform
                guess.append( (p.prior.high + p.prior.low) / 2.0 )

            # Handles scipy bounds safely
            low_b = None if np.isinf(p.prior.low) else p.prior.low
            high_b = None if np.isinf(p.prior.high) else p.prior.high
            bounds.append( (low_b, high_b) )

            # extract the proposed scale to help the searcher
            scale = getattr(p, "proposed_scale", 0.01)
            scales.append(scale)

        return np.array(guess), bounds, np.array(scales)
    
    def _find_valid_spawn(self, initial_guess, scales, bounds, objective):
        """
        Throws random darts around the initial guess until it finds a physically
        valid universe (chi2 < 1e5). Prevents the optimizer from suffocating in penalty zones.
        """
        # check if the exact center is already valid
        if objective(initial_guess) < 1e5:
            return initial_guess
        else:
            if self.verbose:
                print("[SamplerBase] Initial guess is unphysical. Initiating the Valid Spawn search . . .")

            # Extract lower and upper bounds, safely handling None (unbounded)
            lower_bnds = [b[0] if b[0] is not None else -np.inf for b in bounds]
            upper_bnds = [b[1] if b[1] is not None else np.inf for b in bounds]

            for i in range(1000):
                # jitter the parameters based on their proposed scales
                inflation_factor = 1.0 + (i / 10.0)
                test_guess = initial_guess + np.random.randn(self.ndim) * (scales * inflation_factor)
                test_guess = np.clip(test_guess, lower_bnds, upper_bnds)
                if objective(test_guess) < 1e5:
                    if self.verbose:
                        print(f"[SamplerBase] Valid spawn point found after {i+1} attempts.")
                    return test_guess
                
            raise RuntimeError("[SamplerBase] Failed to find a physically valid starting point after 1000 attempts. Check your priors")
        
        
    def optimize_starting_position(self):
        """
        Runs a global pre-Flight optimizer to find the MAP.
        This guarantees the walkers start exactly at the bottom of the valley.
        """
        if self.verbose:
            print("[SamplerBase] Running Pre-Flight Optimizer to map the probability valley . . .")
        raw_guess, bounds, scales = self._get_dynamical_guess_and_bounds() 

        # optimizer objective : minimize negative log-posteriors
        def objective(theta):
            lp = self.lnpost(theta)
            return -lp if np.isfinite(lp) else 1e10
        
        safe_guess = self._find_valid_spawn(raw_guess, scales, bounds, objective)

        # Shift the objective to be zero at the starting point so that ftol
        # remains a meaningful convergence criterion regardless of how large the
        # log-likelihood normalization constants are (e.g. Pantheon+ stat+sys
        # covariance inflates lnpost by ~65000).
        #
        # Method: Powell (not Nelder-Mead).
        #   Nelder-Mead converges only when BOTH xatol AND fatol are satisfied
        #   simultaneously.  When any parameter is unconstrained by the active
        #   likelihoods (e.g. lambda0 in fQ_Hybrid with background-only data),
        #   the posterior is flat in that direction.  The simplex never collapses
        #   there → xatol is never met → Nelder-Mead exhausts all maxiter=3000
        #   iterations, calling lnposterior ~6000 times (minutes of wall time).
        #
        #   Powell does line minimization along conjugate directions.  In a flat
        #   direction the line search returns in 1–3 evaluations (any point is a
        #   minimum), and Powell simply moves on.  It converges the remaining
        #   active parameters in ~200–500 total evaluations instead of ~6000.
        f0 = objective(safe_guess)
        def shifted_objective(theta):
            return objective(theta) - f0

        res = minimize(
            shifted_objective,
            safe_guess,
            method='Powell',
            bounds=bounds,
            options={'ftol': 1e-4, 'xtol': 1e-4, 'maxiter': 500, 'maxfev': 3000}
        )

        # res.fun is the shifted value; restore absolute for reporting
        abs_fun = res.fun + f0
        if res.success or abs_fun < 1e5:
            if self.verbose:
                print(f"[SamplerBase] Optimizer found the MAP. Best-fit -2*ln(posterior): {abs_fun*2:.2f}")
            return res.x
        else:
            if self.verbose:
                print("[SamplerBase] WARNING!!! Optimizer struggling. Walkers will spawn at the Valid Spawn point.")
            return safe_guess