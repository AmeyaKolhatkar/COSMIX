"""MetropolisHastings — minimal random-walk Metropolis-Hastings sampler.

A bare-bones MCMC sampler using symmetric Gaussian proposals.
Useful for low-dimensional testing and debugging because it has no
external dependencies.  For production runs prefer emcee (affine
invariant) or the nested samplers.

Usage: set sampler_name = "mh" in input.yaml.
"""
import numpy as np
from tqdm import trange

class MHSampler:
    """ 
    Minimal Metropolis-Hastings Sampler
    """
    def __init__(self, lnpost, ndim, proposal_scale=0.1, rng=None):
        """
        lnpost: callable; lnposterior: lnpost(theta) -> float
        ndim: int; number of free parameters
        proposal_scale: float/array of floats; S.D. of Gaussian proposal
        rng: np.random.Generator or None
        """
        self.lnpost = lnpost
        self.ndim= ndim
        self.proposal_scale = proposal_scale
        self.rng = rng or np.random.default_rng()

    def run(self, theta0, nsteps):
        """
        Parameters:

        theta0: array_like; Initial theta
        nsteps: int; number of MCMC steps

        Returns:

        samples: ndarray; shape -> (nsteps, ndim)
        lnprobs: ndarray; shape -> (nsteps,)
        accept_rate: float
        """
        theta = np.asarray(theta0, dtype=float)
        lnP = self.lnpost(theta)

        samples = np.zeros((nsteps, self.ndim))
        lnprobs = np.zeros(nsteps)

        accepted = 0.0

        for i in trange(nsteps, desc="MH Sampling"):
            # proposal
            step = self.rng.normal(scale=self.proposal_scale, size=self.ndim)
            theta_proposed = theta + step
            lnP_proposed = self.lnpost(theta_proposed)

            # accept/reject logic
            if np.isfinite(lnP_proposed):
                delta = lnP_proposed - lnP
                if delta >= 0 or np.log(self.rng.random()) < delta:
                    theta = theta_proposed
                    lnP = lnP_proposed
                    accepted += 1

            samples[i] = theta
            lnprobs[i] = lnP

        accept_rate = accepted/nsteps

        return samples, lnprobs, accept_rate
