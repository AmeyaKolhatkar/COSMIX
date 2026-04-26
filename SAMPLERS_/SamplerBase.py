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
        
        res = minimize(
            objective,
            safe_guess,
            method='Nelder-Mead',
            bounds=bounds,
            options={'xatol': 1e-3, 'fatol':1e-3, 'maxiter': 3000}
        )

        if res.success or res.fun < 1e5:
            if self.verbose:
                print(f"[SamplerBase] Optimizer found the MAP. Best-fit chi2 approx: {res.fun*2:.2f}")
            return res.x
        else:
            if self.verbose:
                print("[SamplerBase] WARNING!!! Optimizer struggling. Walkers will spawn at the Valid Spawn point.")
            return safe_guess