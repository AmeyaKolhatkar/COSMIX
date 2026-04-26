"""GrowthKinematics — linear matter growth engine.

Solves the linear growth equation

    δ'' + (2 - q) δ' - 1.5 μ_G(z) Ω_m(z) δ = 0

using the GrowthSolver (RK4 by default).  The background H(z) comes from
a pre-computed BackgroundKinematics instance so no re-integration is needed.

Exposes the capabilities expected by fill_theory_cache:
    delta(z)    — normalised linear growth factor D(z)/D(0)
    f(z)        — logarithmic growth rate d ln D / d ln a
    sigma8(z)   — σ₈(z) = σ₈₀ × delta(z)
    fsigma8(z)  — f(z) × σ₈(z)  (measured by RSD surveys)
"""
import numpy as np
from CORE_.ObservableEngineBase import ObservableEngineBase
from THEORY_.Solvers_.ODESolver import ODESolver
from THEORY_.Solvers_.GrowthSolver import GrowthSolver


class GrowthKinematics(ObservableEngineBase):
    """
    Linear Growth Engine
    """
    capabilities = {
        "delta", "f", "sigma8", "fsigma8"
    }

    def __init__(self, background, model, theta):
        self.bg = background
        self.model = model
        self.theta = theta
        # sigma80 is only needed for sigma8/fsigma8 observables; defer lookup so
        # runs that only request f or delta (e.g. EgStatistic) don't require it.
        self._sigma80 = None

        self._solve_growth()

    def _solve_growth(self):
        solver = GrowthSolver(
            background=self.bg,
            model=self.model,
            theta=self.theta,
            ode_solver=ODESolver()
        )

        solver.solve()

        self._solver = solver

    def delta(self, z):
        return self._solver.delta(z)

    def f(self, z):
        return self._solver.f(z)
    
    def _get_sigma80(self):
        if self._sigma80 is None:
            self._sigma80 = self.model.pm.get_value(self.theta, "sigma80")
        return self._sigma80

    def sigma8(self, z):
        return self._get_sigma80() * self.delta(z)

    def fsigma8(self, z):
        return self.f(z) * self.sigma8(z)