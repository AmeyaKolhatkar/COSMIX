# ODE Solver

import numpy as np
from scipy.integrate import solve_ivp

class ODESolver:
    """
    Thin wrapper around scipy.integrate.solve_ivp

    Physics agnostic
    """

    def __init__(self, method="RK45", rtol=1e-6, atol=1e-9, max_step=np.inf):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

    def solve(self, rhs, x_span, y0, x_eval, theta=None):
        """
        Inputs:
            - rhs: callable; rhs(x, y, theta) --> dy/dx
            - x_span: (x0, x1)
            - y0: array-like
            - x_eval: grid where solution is needed
            - theta: parameter vector passed through

        returns:

        dict with keys:
            x
            y   (shape: nvar x npts)
        """
        def wrapped_rhs(x, y):
            return rhs(x, y, theta)
        
        sol = solve_ivp(
            wrapped_rhs,
            t_span=x_span,
            y0=np.asarray(y0, dtype=float),
            t_eval=np.asarray(x_eval, dtype=float),
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step
        )

        if not sol.success:
            raise RuntimeError(f"ODE solve failed; {sol.message}")
        
        return {
            "x": sol.t,
            "y": sol.y
        }