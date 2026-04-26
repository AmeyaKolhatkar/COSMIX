# Polynomial Model f(Q)

import numpy as np
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.TheoryCache import TheoryCache
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from THEORY_.Solvers_.ODESolver import ODESolver

from Constants import c, Omegar0

class fQPoly(CosmologyModelBase):
    """
    f(Q) = Q + alpha (Q/Q_0)**n
    """
    name = "fQ_Poly"

    def __init__(self, pm):
        super().__init__(pm)

    def solve_background_numerically(self, theta, z_grid, config):
        solver = ODESolver(method="BDF")        # BDF -> stiff-safe
        y0 = [...]

        def rhs(z, y, theta):
            ... = y 
            H = self._H_from_densities(..., theta)

            drho_m = ...
            drho_DE = ...

            return [drho_m, drho_DE]
        
        sol = solver.solve(
            rhs=rhs,
            x_span=(z_grid[0], z_grid[-1]),
            y0=y0,
            x_eval=z_grid,
            theta=theta
        )

        rho_m, rho_de = sol["y"]
        H = self._H_from_densities(rho_m, rho_de, theta)

        return {"H": H}
    
    def _H(self, z, H0, Omegam0, alpha, n):
        ...
        ...
        ...
        

