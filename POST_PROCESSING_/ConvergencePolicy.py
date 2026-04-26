# Convergence Policy
"""
Responsibilities
    - Define what convergence means
    - Combines ESS, Tau and rhat coherently
    - Produce a machine readable verdict
"""

import numpy as np

class ConvergencePolicy:
    def __init__(self, ess_min=1000, tau_factor=50, rhat_tol=0.01, require_multi=True):
        self.ess_min = ess_min
        self.tau_factor = tau_factor
        self.rhat_tol = rhat_tol
        self.require_multi = require_multi

    def evaluate(self, results):
        """
        returns:
        dict with keys:
            - converged (bool)
            - checks (dict)
        """

        checks = {}

        #----- ESS -----#
        if results.ess is None:
            checks["ess"] = False
        else:
            checks["ess"] = np.all(results.ess > self.ess_min)

        #----- Tau -----#
        if results.tau is None:
            checks["tau"] = False
        else:
            n = results.chain.shape[0]
            checks["tau"] = np.all(n > self.tau_factor * results.tau)

        #----- rhat (if available) -----#
        if hasattr(results, "rhat"):
            checks["rhat"] = np.all(results.rhat - 1 < self.rhat_tol)
        else:
            checks["rhat"] = not self.require_multi

        converged = all(checks.values())

        return {
            "converged": bool(converged),
            "checks": checks
        }