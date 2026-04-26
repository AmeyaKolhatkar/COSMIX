# Root Solver
import numpy as np
from scipy.optimize import root_scalar

class RootSolver:
    """
    Physics-agnostic solver for implicit algebraic equations F(y, x, theta) = 0
    """
    def __init__(self, method='brentq', xtol=1e-8, rtol=1e-8, maxiter=100):
        self.method = method
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter

    def solve(self, implicit_eq, x_eval, bracket, theta=None, y0_guess=None):
        """
        Inputs:
            - implicit_eq   : callable;     implicit_eq(y, x, theta) --> float
            - x_eval        : array-like;   the grid (e.g. redshift) where y is needed
            - bracket       : tuple (a, b); search interval for the root (required for brentq)
            - theta         : parameter vector
            - y0_guess      : array-like;   initial guesses if using methods like Newton/fsolve

        Returns:
            - dict with keys: 'x' and 'y'
        """
        x_eval = np.asarray(x_eval, dtype=float)
        y_sol = np.zeros_like(x_eval)

        # iterate over the grid to find roots.
        # For high performance, this loop should be compiled eventually with Numba or JAX

        for i, x_val in enumerate(x_eval):
            def objective(y):
                return implicit_eq(y, x_val, theta)
            
            try:
                # Brent's method is the most robust for cosmological root finding.
                sol = root_scalar(
                    objective,
                    bracket=bracket,
                    method=self.method,
                    xtol=self.xtol,
                    rtol=self.rtol,
                    maxiter=self.maxiter
                )
                if not sol.converged:
                    raise RuntimeError(f"RootSolver failed to converge at x={x_val}")
                
                y_sol[i] = sol.root

            except Exception as e:
                raise RuntimeError(f"RootSolver failed at x={x_val}: {str(e)}")
            
        return {
            "x": x_eval,
            "y": y_sol
        }