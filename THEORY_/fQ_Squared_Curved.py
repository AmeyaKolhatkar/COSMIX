
"""fQ_Squared_Curved — f(Q) quadratic model on the unique curved FLRW connection.

f(Q) = alpha1·Q + alpha2·Q₀ + alpha3·Q²

where Q₀ is the non-metricity scalar today (computed from eq. 77 at z=0).

Parametrisation
---------------
Instead of sampling alpha3 directly — which has a physically valid range of
O(1/Q₀²) ~ O(10⁻⁹) while its natural prior would be O(1) — we use the
dimensionless coupling:

    β₃ = alpha3 · Q₀         prior: UniformPrior(−0.49, 0.49)

so that  f′(Q₀) = alpha1 + 2·beta3  stays manifestly positive for the whole
prior (prior boundary > −0.5 ensures f′(Q₀) > 0.02 > 0).

alpha2 is NOT a free parameter.  It is derived from the Friedmann constraint
at z = 0 (eq. 78a with P₀ = 0), which gives:

    alpha2 = [ 2ρ₀  −  6·f′(Q₀)·(H₀² + k)  +  alpha3·Q₀² ] / Q₀

where  ρ₀ = 3H₀²(Ωm₀ + Ωr₀),  k = −Ωk·H₀².
This eliminates one free parameter and enforces H(z=0) = H₀ exactly.

Free parameters: H0, Omegam0, Omegak, gamma_0  (base class) + beta3 (5 total)
Fixed:           alpha1 = 1
Derived:         alpha2, alpha3 = beta3 / Q₀
"""
import numpy as np
from THEORY_.CurvedfQBase import CurvedfQBase
from CORE_.ParameterManager_ import Parameter, UniformPrior
from Constants import Omegar0


class fQSquaredCurved(CurvedfQBase):
    """
    f(Q) = alpha1·Q + alpha2·Q₀ + alpha3·Q²

    Inherits the curved FLRW ODE background solver from CurvedfQBase.
    Model-specific parameters (alpha1, alpha2, alpha3, Q₀) are cached as
    instance variables before each background solve and physicality check
    so that f, f_prime, f_double_prime, f_triple_prime can access them.
    """
    name = "fQ_Squared_Curved"

    def __init__(self, pm):
        super().__init__(pm)
        # Model-specific parameters; set via _set_params_from_theta before
        # any f(Q) evaluation.  Safe for COSMIX's single-threaded MCMC usage.
        self._alpha1 = None
        self._alpha2 = None
        self._alpha3 = None
        self._Q0     = None

    # ── Internal helper ──────────────────────────────────────────────────────
    def _set_params_from_theta(self, theta):
        """Cache model-specific parameters from theta before any f(Q) call.

        Computes:
            self._alpha3  = beta3 / Q₀          (physical quadratic coeff.)
            self._alpha2  = derived from Friedmann at z=0  (ensures H(0)=H₀)
            self._alpha1  = fixed (from theta, typically 1)
            self._Q0      = Q at z=0 from kinematic IC (eq. 77)
        """
        H0      = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        Omegak  = self.pm.get_value(theta, "Omegak")
        gamma_0 = self.pm.get_value(theta, "gamma_0")

        self._alpha1 = self.pm.get_value(theta, "alpha1")
        beta3        = self.pm.get_value(theta, "beta3")

        # ── Q₀ from kinematic IC (eq. 77 at z=0, γ̇₀=0) ─────────────────────
        try:
            y0 = self._initial_conditions(H0, Omegam0, Omegak, gamma_0)
            self._Q0 = y0[0]
        except (ValueError, RuntimeError):
            self._Q0 = -6.0 * H0**2   # flat-space fallback

        # ── α₃ from dimensionless coupling β₃ = α₃·Q₀ ───────────────────────
        # Guard against degenerate Q₀ (only possible if gamma_0 is very large
        # in a contrived configuration; practically never reached in prior).
        if abs(self._Q0) < 1.0:
            self._alpha3 = 0.0
        else:
            self._alpha3 = beta3 / self._Q0

        # ── α₂ from Friedmann self-consistency at z=0 (P₀=0, B=0) ───────────
        # 3·f′(Q₀)·(H₀²+k) + ½(α₂·Q₀ − α₃·Q₀²) = ρ₀
        # → α₂ = [ 2ρ₀ − 6·f′(Q₀)·(H₀²+k) + α₃·Q₀² ] / Q₀
        k    = -Omegak * H0**2
        rho0 = 3.0 * H0**2 * (Omegam0 + Omegar0)
        fp0  = self._alpha1 + 2.0 * self._alpha3 * self._Q0   # = alpha1 + 2·beta3
        if abs(self._Q0) < 1.0:
            self._alpha2 = 0.0
        else:
            self._alpha2 = (
                2.0 * rho0
                - 6.0 * fp0 * (H0**2 + k)
                + self._alpha3 * self._Q0**2
            ) / self._Q0

    # ── Abstract interface ───────────────────────────────────────────────────
    def f(self, Q):
        return self._alpha1 * Q + self._alpha2 * self._Q0 + self._alpha3 * Q**2

    def f_prime(self, Q):
        return self._alpha1 + 2.0 * self._alpha3 * Q

    def f_double_prime(self, Q):
        return 2.0 * self._alpha3

    def f_triple_prime(self, Q):
        return 0.0

    # ── Parameters ───────────────────────────────────────────────────────────
    @classmethod
    def declare_parameters(cls):
        base = super().declare_parameters()   # H0, Omegam0, Omegak, gamma_0
        return base + [
            Parameter(
                name="alpha1",
                latex=r'\alpha_1',
                prior=UniformPrior(low=0.0, high=5.0),
                role="cosmo",
                status="fixed",
                value=1.0,
                proposed_scale=0.05
            ),
            # beta3 = alpha3 · Q₀  (dimensionless quadratic coupling).
            # f′(Q₀) = alpha1 + 2·beta3, so the prior boundary (±0.49)
            # keeps f′(Q₀) ∈ (0.02, 1.98) — always positive, no ghost at z=0.
            # alpha2 is derived from the Friedmann constraint; not sampled.
            Parameter(
                name="beta3",
                latex=r'\beta_3',
                prior=UniformPrior(low=-0.49, high=0.49),
                role="cosmo",
                status="free",
                proposed_scale=0.02
            ),
        ]