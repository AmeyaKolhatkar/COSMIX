"""NestedSamplingBase — shared infrastructure for nested samplers.

Inherits from SamplerBase and overrides MCMC-specific methods so that
Dynesty, PolyChord, and any future nested sampler only need to implement
their own ``run()`` method.  Separates the lnlike and prior-transform
evaluations which nested sampling handles differently from MCMC.
"""

import numpy as np
from SAMPLERS_.SamplerBase import SamplerBase

class NestedSamplerBase(SamplerBase):
    """
    Base Class for Nested Sampling samplers like Dynesty, MultiNest, PolyChord, etc.
    Overrides MCMC-specific methods and attributes from SamplerBase to implement nested sampling logic.
    Separates the lnlike and prior evaluation.
    """
    def __init__(self, pm, pipeline):
        super().__init__(pm, pipeline.lnposterior)       # the lnpost is dummy since Nested Sampling does not use it.
        self.pipeline = pipeline

    def loglike(self, theta):
        """
        Nested Sampling Requires pure log-likelihood without the prior since prior is already 
        accounted for by prior transforms.  
        """
        # recheck physical bounds
        if not np.isfinite(self.pm.lnprior(theta)):
            return -np.inf
        
        try:
            lnlike = self.pipeline.lnlike(theta)
            return lnlike if np.isfinite(lnlike) else -np.inf
        except Exception as e:
            return -np.inf
        
    def wrapper_prior_transform(self, cube):
        """
        Wrapper for the prior transform method in ParameterManager.
        """
        return self.pm.prior_transform(cube)
