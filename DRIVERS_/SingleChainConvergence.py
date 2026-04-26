"""SingleChainConvergence — convergence wrappers for single-chain runs.

Provides two strategies:

SingleChainStrategy
    Runs a single MCMC chain (emcee or MH).  Convergence is assessed via
    the integrated autocorrelation time τ and effective sample size.
    Use this for quick tests or when only one core is available.

NestedStrategy
    Wraps a nested sampler (Dynesty or PolyChord).  Nested sampling is
    self-terminating so there is no external convergence criterion —
    the algorithm stops automatically.  The logZ uncertainty is reported
    as the quality metric instead of R̂ or τ.
"""
from DRIVERS_.ConvergenceStrategy import ConvergenceStrategy
from POST_PROCESSING_.ResultsContainer import MCMCResults
from POST_PROCESSING_.Diagnostics import MCMCDiagnostics

class SingleChainStrategy(ConvergenceStrategy):
    def __init__(self, sampler, pipeline, run_kwargs):
        self.sampler = sampler
        self.pipeline = pipeline
        self.run_kwargs = run_kwargs
        self.results = None

    def run(self):
        raw = self.sampler.run(**self.run_kwargs)
        self.results = MCMCResults.from_sampler_output(
            results=raw,
            pipeline=self.pipeline,
            sampler_name=self.sampler.__class__.__name__
        )
        
        return self.results
    
    def is_converged(self):
        diag = MCMCDiagnostics(self.results)

        return diag.is_converged_single()
    
    def summary(self):
        return {
            "mode": "single",
            "converged": self.is_converged(),
            "ess": self.results.ess,
            "tau": self.results.tau
        }


class NestedStrategy(ConvergenceStrategy):
    """
    Convergence strategy for nested samplers (Dynesty, PolyChord, MultiNest).

    Nested sampling is self-terminating: it stops when the remaining prior
    volume contributes negligible evidence (controlled by precision_criterion).
    There is no R-hat or τ concept — convergence is guaranteed by the
    algorithm itself once it finishes.  We simply run it once and report the
    logZ uncertainty as the quality metric.
    """
    def __init__(self, sampler, pipeline):
        self.sampler = sampler
        self.pipeline = pipeline
        self.results = None
        self._raw = None

    def run(self):
        self._raw = self.sampler.run()
        self.results = MCMCResults.from_nested_output(
            results=self._raw,
            pipeline=self.pipeline,
            sampler_name=self.sampler.__class__.__name__
        )
        return self.results

    def is_converged(self):
        # Nested sampling is converged by construction when it finishes.
        return True

    def summary(self):
        return {
            "mode": "nested",
            "converged": True,
            "sampler": self.sampler.__class__.__name__,
            "logZ": self._raw.get("logZ") if self._raw else None,
            "logZ_err": self._raw.get("logZ_err") if self._raw else None,
        }
