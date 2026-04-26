# Background Problem Descriptors
"""
Strategy pattern for background cosmology solves.

Each model returns one of these from background_problem(theta, z_grid).
BackgroundKinematics calls problem.solve() without knowing how the physics works.

Contract: .solve() returns a dict with at least {"H": np.ndarray}.
Optional keys: "Omegam", or any other derived quantity the model wants to pass through.
"""

import numpy as np
from THEORY_.Solvers_.RK4Solver import numba_rk4


class AnalyticalProblem:
    """
    The model already has a closed-form H(z).
    """
    __slots__ = ("_h_func", "_extras_func")

    def __init__(self, h_func, extras_func=None):
        """
        h_func      : callable(z_grid) -> H array
        extras_func : callable(z_grid) -> dict of additional arrays (e.g. {"Omegam": ...})
        """
        self._h_func = h_func
        self._extras_func = extras_func

    def solve(self, z_grid):
        result = {"H": self._h_func(z_grid)}
        if self._extras_func is not None:
            result.update(self._extras_func(z_grid))
        return result


class NumbaODEProblem:
    """
    The model needs to integrate a Numba-compiled ODE to get H(z).
    """
    __slots__ = ("_rhs", "_y0", "_params", "_extract")

    def __init__(self, rhs, y0, params, extract):
        """
        rhs     : @njit function(x, y, params) -> dy/dx
        y0      : np.ndarray initial state at z=0
        params  : np.ndarray packed parameter array for the jitted RHS
        extract : callable(sol_y, z_grid) -> dict with at least {"H": ...}
        """
        self._rhs = rhs
        self._y0 = y0
        self._params = params
        self._extract = extract

    def solve(self, z_grid):
        try:
            sol_y = numba_rk4(self._rhs, z_grid, self._y0, self._params)
        except ValueError as e:
            raise RuntimeError(str(e))
        return self._extract(sol_y, z_grid)


class ImplicitProblem:
    """
    The model defines an implicit algebraic equation F(y, z, params) = 0
    that must be root-solved at each redshift point.
    """
    __slots__ = ("_equation", "_bracket", "_extract", "_xtol")

    def __init__(self, equation, bracket, extract, xtol=1e-8):
        """
        equation : callable(y, z, params) -> float  (the residual)
        bracket  : (a, b) search interval for root_scalar
        extract  : callable(y_array, z_grid) -> dict with at least {"H": ...}
        xtol     : absolute tolerance for the root finder
        """
        from scipy.optimize import root_scalar
        self._equation = equation
        self._bracket = bracket
        self._extract = extract
        self._xtol = xtol

    def solve(self, z_grid):
        from scipy.optimize import root_scalar

        y_sol = np.empty(len(z_grid), dtype=np.float64)

        for i, z_val in enumerate(z_grid):
            sol = root_scalar(
                lambda y: self._equation(y, z_val),
                bracket=self._bracket,
                method='brentq',
                xtol=self._xtol,
                rtol=1e-8,
                maxiter=100
            )
            if not sol.converged:
                raise RuntimeError(f"ImplicitProblem: root finder failed at z={z_val:.4f}")
            y_sol[i] = sol.root

        return self._extract(y_sol, z_grid)
