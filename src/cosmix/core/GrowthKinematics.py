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
from cosmix.core.ObservableEngineBase import ObservableEngineBase
from cosmix.theory.Solvers_.ODESolver import ODESolver
from cosmix.theory.Solvers_.GrowthSolver import GrowthSolver


class GrowthKinematics(ObservableEngineBase):
    """
    Linear Growth Engine
    """
    capabilities = {
        "delta", "f", "sigma8", "fsigma8", "growth_ratio"
    }

    def __init__(self, background, model, theta):
        self.bg = background
        self.model = model
        self.theta = theta
        # sigma80 is only needed for sigma8/fsigma8 observables; defer lookup so
        # runs that only request f or delta (e.g. EgStatistic) don't require it.
        self._sigma80 = None
        # ΛCDM reference growth amplitude, solved lazily on first growth_ratio
        # call and reused thereafter (see growth_ratio).
        self._D0_ref = None

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

    # ── Implied-A_s diagnostic ──────────────────────────────────────────────
    def growth_ratio(self, z):
        """R = D0_model / D0_ΛCDM — the total-growth ratio at z=0.

        Both amplitudes are the *unnormalised* growth D(a=1) obtained by
        integrating the same ODE, from the same N_ini = -7 (z ≈ 1096), with
        the same δ ∝ a initial condition — the model with its own H(z) and
        μ_G(z), the reference with a ΛCDM H(z) at the same Ω_m0 and μ_G ≡ 1.

        Why this is the right object.  The implied primordial amplitude
        follows from σ₈₀² = A_s · 𝒥(θ) · D₀², where 𝒥 is the transfer/window
        integral.  A Boltzmann code run for ΛCDM at a reference amplitude
        gives σ₈,ref² = A_s^ref · 𝒥(θ) · D₀,ΛCDM², so dividing the two
        cancels 𝒥 exactly and leaves

            A_s^implied = A_s^ref (σ₈₀ / σ₈,ref)² / R² .

        The cancellation is legitimate because everything earlier than
        N_ini is common to model and reference: by z ≈ 1100 the coincident
        f(Q) models here have μ_G → 1 and (for the Hybrid) the Q₀²/Q term
        → 0, so the transfer function is ΛCDM-identical.  R therefore
        carries the entire late-time growth modification and nothing else.

        Scalar-valued; returned broadcast over `z` because TheoryCache
        stores arrays keyed on a redshift grid.

        Returns None if the reference integration fails, which marks the
        theory cache invalid (see fill_theory_cache).
        """
        import numpy as np
        from cosmix.core.BackgroundKinematics import BackgroundKinematics
        from cosmix.core.BackgroundConfiguration import BackgroundConfig
        from cosmix.Constants import Omegar0

        D0_model = getattr(self._solver, "unnorm_D0", None)
        if D0_model is None or not np.isfinite(D0_model) or D0_model <= 0:
            return None

        if self._D0_ref is None:
            Om0 = self.model.pm.get_value(self.theta, "Omegam0")
            OL0 = 1.0 - Om0 - Omegar0

            # Minimal ΛCDM stand-in for the reference growth: only the four
            # accessors GrowthSolver.solve() reads are needed.
            class _LCDMRefBG:
                def __init__(self, zg, Om0):
                    self._zg, self.Om0 = zg, Om0
                def z_grid(self):
                    return self._zg
                def E(self, z):
                    z = np.asarray(z, dtype=float)
                    return np.sqrt(self.Om0 * (1 + z) ** 3
                                   + Omegar0 * (1 + z) ** 4 + OL0)
                def dlnH_dN(self, z):
                    a = 1.0 / (1.0 + np.asarray(z, dtype=float)); h = 1e-5
                    lnE = lambda aa: np.log(self.E(1.0 / aa - 1.0))
                    return (lnE(a * np.exp(h)) - lnE(a * np.exp(-h))) / (2 * h)
                def Omegamz(self, z):
                    z = np.asarray(z, dtype=float)
                    return self.Om0 * (1 + z) ** 3 / self.E(z) ** 2
                def muG(self, z):
                    return np.ones_like(np.asarray(z, dtype=float))

            try:
                ref = GrowthSolver(
                    background=_LCDMRefBG(self.bg.z_grid(), Om0),
                    model=None, theta=None, ode_solver=ODESolver(),
                )
                ref.solve()
                self._D0_ref = ref.unnorm_D0
            except Exception:
                return None

        if not np.isfinite(self._D0_ref) or self._D0_ref <= 0:
            return None

        return np.full_like(np.asarray(z, dtype=float),
                            D0_model / self._D0_ref)