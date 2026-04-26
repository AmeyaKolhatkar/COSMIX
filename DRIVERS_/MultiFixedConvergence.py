"""MultiFixedConvergence — run N chains once and report R̂/ESS/τ.

Runs MultiChainDriver.run() a single time (all chains in parallel),
then checks convergence using the Gelman-Rubin R̂ statistic, the
effective sample size (ESS), and the integrated autocorrelation time.

This is the simplest multi-chain strategy — use it when you already
know roughly how many steps are needed.  For automatic step-doubling
until convergence, use MultiAutoConvergence instead.
"""
from DRIVERS_.ConvergenceStrategy import ConvergenceStrategy

class MultiFixedStrategy(ConvergenceStrategy):
    def __init__(self, driver, run_kwargs, rhat_tol=0.01, ess_min=1000, tau_factor=50):
        self.driver = driver
        self.run_kwargs = run_kwargs
        self.rhat_tol = rhat_tol
        self.ess_min = ess_min
        self.tau_factor = tau_factor
        self.results = None

    def run(self):
        self.results = self.driver.run(self.run_kwargs)

        return self.results
    
    def is_converged(self):
        return self.results.is_converged(
            tol=self.rhat_tol,
            ess_min=self.ess_min,
            tau_factor=self.tau_factor
        )
    
    def summary(self):
        detail = self.results.convergence_detail(
            tol=self.rhat_tol,
            ess_min=self.ess_min,
            tau_factor=self.tau_factor
        )
        return {
            "mode": "multi_fixed",
            "converged": self.is_converged(),
            "nchains": self.driver.nchains,
            "rhat": self.results.rhat.tolist(),
            "tau": self.results.tau.tolist(),
            "ess": self.results.ess.tolist(),
            "detail": detail
        }