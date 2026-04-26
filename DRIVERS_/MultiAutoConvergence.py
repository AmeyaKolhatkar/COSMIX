"""MultiAutoConvergence — Cobaya-style continuous run until convergence.

Chains start once (single Pre-Flight Optimizer call per chain) and run
indefinitely in check_every-step intervals.  After each interval all
accumulated samples are pooled and R̂, ESS, and τ are evaluated.  The run
stops when all criteria are satisfied or max_steps is reached.

Key parameters
--------------
check_every : int
    Steps per check interval.  Small values (200–500) give more frequent
    diagnostics; large values reduce overhead.
max_steps : int
    Hard upper bound on the total number of steps per chain.

Replaces the old chunk-based auto_run approach (min_chunks / max_chunks),
which restarted chains from scratch each chunk.
"""
from DRIVERS_.ConvergenceStrategy import ConvergenceStrategy

class MultiAutoStrategy(ConvergenceStrategy):
    def __init__(self, driver, run_kwargs, rhat_tol=0.01, ess_min=1000,
                 tau_factor=50, check_every=500, max_steps=50000):
        self.driver      = driver
        self.run_kwargs  = run_kwargs
        self.rhat_tol    = rhat_tol
        self.ess_min     = ess_min
        self.tau_factor  = tau_factor
        self.check_every = check_every
        self.max_steps   = max_steps
        self.results     = None
        self.status      = None

    def run(self):
        self.results, self.status = self.driver.continuous_auto_run(
            run_kwargs   = self.run_kwargs,
            rhat_tol     = self.rhat_tol,
            ess_min      = self.ess_min,
            tau_factor   = self.tau_factor,
            check_every  = self.check_every,
            max_steps    = self.max_steps,
        )
        return self.results

    def is_converged(self):
        return self.status["status"] == "converged"

    def summary(self):
        detail = self.results.convergence_detail(
            tol        = self.rhat_tol,
            ess_min    = self.ess_min,
            tau_factor = self.tau_factor,
        )
        return {
            "mode":        "multi_auto",
            "converged":   self.is_converged(),
            "nchains":     self.driver.nchains,
            "total_steps": self.status["total_steps"],
            "iterations":  self.status["iterations"],
            "rhat":        self.results.rhat.tolist(),
            "tau":         self.results.tau.tolist(),
            "ess":         self.results.ess.tolist(),
            "rhat_tol":    self.rhat_tol,
            "ess_min":     self.ess_min,
            "tau_factor":  self.tau_factor,
            "detail":      detail,
        }
