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
    def __init__(self, pm, pipeline, nlive=500, n_cpu=1, sample='rslice'):
        super().__init__(pm, pipeline)
        self.nlive = nlive
        self.n_cpu = max(1, int(n_cpu))
        self.sample = sample

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
                    sample=self.sample,
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
                sample=self.sample,
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

        # Return ALL dead points with their importance weights — no resampling.
        # Reference standard (Cobaya/PolyChord): each sample carries weight=exp(logwt),
        # and MCSamples(weights=...) passes them through to GetDist's weighted KDE.
        # Resampling (bootstrap or resample_equal) maps N_dead→N_dead with massive
        # duplication when ESS<<N_dead, creating point clusters that produce jagged
        # contours regardless of nlive. Weighted samples avoid this entirely.
        chain    = res.samples   # (N_dead, ndim)
        log_prob = res.logl      # (N_dead,) — pure log-likelihoods

        best_idx = int(np.argmax(log_prob))
        best_fit = chain[best_idx]

        print(f"[DynestySampler] Sampling complete. log Evidence (logZ): {logZ_physical:.3f} +/- {logZ_err:.3f}")
        print(f"[DynestySampler] Dead points: {len(res.samples)}  ESS: {ess:.0f}")

        # weights=None signals equal-weight samples.  DatasetConsistency computes
        # KL divergence as a simple mean — unbiased when the chain is a bootstrap
        # posterior sample, since E_posterior[logl - logZ] = D_KL exactly.
        return {
            "chain"         : chain,
            "log_prob"      : log_prob,
            "weights"       : weights,
            "best_fit"      : best_fit,
            "logZ"          : logZ,
            "logZ_physical" : logZ_physical,
            "logZ_err"      : logZ_err,
            "raw_results"   : res,
        }