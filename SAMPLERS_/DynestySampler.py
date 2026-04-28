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
import dynesty.pool as dypool
from SAMPLERS_.NestedSamplingBase import NestedSamplerBase

class DynestySampler(NestedSamplerBase):
    """
    Wrapper for the pure-Python Dynesty dynamic nested sampler.
    Excellent for highly non-linear problems and multimodal posteriors, but can be slower than C++ implementations
    like MultiNest for high-dimensional problems.
    """
    def __init__(self, pm, pipeline, nlive=500, n_cpu=1):
        super().__init__(pm, pipeline)
        self.nlive = nlive
        self.n_cpu = max(1, int(n_cpu))

    def run(self):
        print("[DynestySampler] Initializing Nested Sampling with Dynesty . . .")

        if self.n_cpu > 1:
            # dynesty.pool.Pool pre-loads loglike and prior_transform into each
            # worker process once at startup.  Only parameter arrays are pickled
            # per call, not the full pipeline/likelihood objects.  This is far
            # faster than plain mp.Pool which re-pickles the bound method on
            # every batch of proposals.
            # Must pass pool.loglike / pool.prior_transform to the sampler —
            # NOT the original self.loglike / self.wrapper_prior_transform.
            print(f"[DynestySampler] Parallelizing across {self.n_cpu} cores via dynesty.pool.Pool.")
            with dypool.Pool(self.n_cpu, self.loglike, self.wrapper_prior_transform) as pool:
                sampler = dynesty.DynamicNestedSampler(
                    loglikelihood=pool.loglike,
                    prior_transform=pool.prior_transform,
                    ndim=self.ndim,
                    bound='multi',
                    sample='rwalk',
                    nlive=self.nlive,
                    pool=pool,
                    queue_size=self.n_cpu,
                )
                print("[DynestySampler] Commencing dynamic nested sampling run . . .")
                sampler.run_nested(print_progress=True)
                res = sampler.results
        else:
            sampler = dynesty.DynamicNestedSampler(
                loglikelihood=self.loglike,
                prior_transform=self.wrapper_prior_transform,
                ndim=self.ndim,
                bound='multi',
                sample='rwalk',
                nlive=self.nlive,
            )
            print("[DynestySampler] Commencing dynamic nested sampling run . . .")
            sampler.run_nested(print_progress=True)
            res = sampler.results

        # Evidence
        logZ          = res.logz[-1]
        logZ_err      = res.logzerr[-1]
        norm          = self.pipeline.norm_terms_total()
        logZ_physical = logZ - norm

        # Importance weights for ALL dead points.
        weights = np.exp(res.logwt - res.logz[-1])
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()
        ess = float(1.0 / np.sum(weights ** 2))

        # Bootstrap resample to equal-weight posterior samples.
        # Draw ≈ ESS samples with replacement proportional to importance weights.
        # Each dead point is selected with frequency ∝ its posterior mass, so the
        # resulting chain is an unweighted posterior sample.  getdist's KDE
        # bandwidth selector sees the correct posterior spread — including tails —
        # giving contour widths that match emcee.  This replaces the old 99.9%-filter
        # approach, which concentrated weight near the peak and shrank contours.
        n_resample = max(int(ess), 500)
        rng = np.random.default_rng()
        idx = rng.choice(len(weights), size=n_resample, replace=True, p=weights)
        chain    = res.samples[idx]
        log_prob = res.logl[idx]

        best_idx = int(np.argmax(log_prob))
        best_fit = chain[best_idx]

        print(f"[DynestySampler] Sampling complete. log Evidence (logZ): {logZ_physical:.3f} +/- {logZ_err:.3f}")
        print(f"[DynestySampler] Dead points: {len(res.samples)}  ESS: {ess:.0f}  Bootstrap posterior: {n_resample} samples")

        # weights=None signals equal-weight samples.  DatasetConsistency computes
        # KL divergence as a simple mean — unbiased when the chain is a bootstrap
        # posterior sample, since E_posterior[logl - logZ] = D_KL exactly.
        return {
            "chain"         : chain,
            "log_prob"      : log_prob,
            "weights"       : None,
            "best_fit"      : best_fit,
            "logZ"          : logZ,
            "logZ_physical" : logZ_physical,
            "logZ_err"      : logZ_err,
            "raw_results"   : res,
        }