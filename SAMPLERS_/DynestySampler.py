"""DynestySampler — Dynesty dynamic nested sampler wrapper.

Dynesty is a pure-Python nested sampler that adapts the number of live
points during the run.  It is well-suited to highly non-linear posteriors
and multimodal distributions.

The sampler returns equal-weighted posterior samples together with
consistently-paired log-likelihood values (chain[i] and log_prob[i]
always correspond to the same parameter point).

Typical usage
-------------
sampler = DynestySampler(pm=pipeline.pm, pipeline=pipeline, nlive=500)
result  = sampler.run()
# result keys: chain, log_prob, best_fit, logZ, logZ_err, raw_results
"""
import numpy as np
import dynesty
from SAMPLERS_.NestedSamplingBase import NestedSamplerBase

class DynestySampler(NestedSamplerBase):
    """
    Wrapper for the pure-Python Dynesty dynamic nested sampler.
    Excellent for highly non-linear problems and multimodal posteriors, but can be slower than C++ implementations
    like MultiNest for high-dimensional problems.
    """
    def __init__(self, pm, pipeline, nlive=500):
        super().__init__(pm, pipeline)
        self.nlive = nlive

    def run(self):
        print("[DynestySampler] Initializing Nested Sampling with Dynesty . . .")
        
        sampler = dynesty.DynamicNestedSampler(
            loglikelihood=self.loglike,
            prior_transform=self.wrapper_prior_transform,
            ndim=self.ndim,
            bound='multi', # Use multi-ellipsoidal bounds for better efficiency in multimodal posteriors
            sample='rwalk', # Use random walk sampling to better explore fractured parameter spaces
            nlive=self.nlive
        )

        print("[DynestySampler] Commencing dynamic nested sampling run . . .")
        sampler.run_nested(print_progress=True)
        res = sampler.results

        # Extract evidence (logZ) and its error
        logZ = res.logz[-1]
        logZ_err = res.logzerr[-1]

        # Subtract likelihood normalization constants so the printed/stored logZ
        # is the physical Bayesian evidence (independent of covariance normalizations).
        norm = self.pipeline.norm_terms_total()
        logZ_physical = logZ - norm

        print(f"[DynestySampler] Sampling complete. log Evidence (logZ): {logZ_physical:.3f} +/- {logZ_err:.3f}")

        # extract equal weighted posterior samples for corner plots and parameter estimation
        weights = np.exp(res.logwt - res.logz[-1])
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()

        # resample both samples and their corresponding log-likelihoods together
        # so that chain[i] and log_prob[i] always refer to the same point
        n = len(weights)
        idx = np.random.choice(n, size=n, replace=True, p=weights)
        samples = res.samples[idx]
        logl    = res.logl[idx]

        best_idx = np.argmax(res.logl)
        best_fit = res.samples[best_idx]

        return {
            "chain": samples,
            "log_prob": logl,
            "best_fit": best_fit,
            "logZ": logZ,
            "logZ_physical": logZ_physical,
            "logZ_err": logZ_err,
            "raw_results": res
        }