"""fQHybrid — f(Q) Hybrid modified gravity model.

Implements the f(Q) = alpha1*Q + alpha2*Q₀ + alpha3*Q₀²/Q model.
The Friedmann equation admits an analytical solution for H(z):

    H(z) = H₀ ( 0.5 (Ω_tilde + sqrt(Ω_tilde² + 4(1-Ω_tilde₀))) )¹²

where  3*alpha3 = alpha1 - Ω₀ - alpha2  (flatness constraint).

Free parameters
---------------
H0       — Hubble constant today
Omegam0  — matter density parameter today
alpha1   — coefficient of the linear Q term
alpha2   — coefficient of the constant Q₀ term
"""
import numpy as np
from cosmix.core.Registry import cosmix_registry
from cosmix.core.CosmologyModelBase import CosmologyModelBase
from cosmix.core.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from cosmix.core.BackgroundConfiguration import BackgroundConfig
from cosmix.theory.Solvers_.BackgroundProblem import AnalyticalProblem
from cosmix.theory.CurvedfQBase import CurvedfQBase 

from cosmix.Constants import c, Omegar0
@cosmix_registry.register_model("fQ_Hybrid")
class fQHybrid(CosmologyModelBase):
    """
    f(Q) = alpha1 Q + alpha2 Q_0 + alpha3 Q_0^2 / Q

    this is the positive branch solution - 
        H(z) = H_0 (0.5 * ( om_tilde + (om_tilde**2 + 4*(1 - om_tilde_0))**0.5 ))**0.5

    3 al3 = al1 - Omega_0 - al2
    """
    name="fQ_Hybrid"

    def __init__(self, pm):
        super().__init__(pm)

    def _H(self, z, H0, Omegam0, alpha1, alpha2, lambda0):
        z = np.asarray(z)
        omegam_tilde = Omegam0 * (1+z)**3 + alpha2 + Omegar0 * (1+z)**4
        omegam_tilde_0 = Omegam0 + alpha2 + Omegar0
        sqrt_arg = omegam_tilde**2 + 4*alpha1*(alpha1-omegam_tilde_0)
        if np.any(sqrt_arg < 0):
            return np.full_like(z, np.nan)
        arg = ( omegam_tilde + np.sqrt(sqrt_arg) ) / (2 * alpha1)
        if np.any(arg < 0):
            return np.full_like(z, np.nan)
        return H0 * np.sqrt(arg)        
    
    def background_problem(self, theta, z_grid):
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha1 = self.pm.get_value(theta, "alpha1")
        alpha2 = self.pm.get_value(theta, "alpha2")
        lambda0 = self.pm.get_value(theta, "lambda0")

        return AnalyticalProblem(
            h_func=lambda z: self._H(z, H0, Omegam0, alpha1, alpha2, lambda0)
        )
    
    def background_config(self):
        return BackgroundConfig(
            z_max=3.0,
            nz=150,
            integration_method="trapz"
        )
    
    def muG(self, z, theta, bg_engine):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha1 = self.pm.get_value(theta, "alpha1")
        alpha2 = self.pm.get_value(theta, "alpha2")
        lambda0 = self.pm.get_value(theta, "lambda0")

        E = bg_engine.E(z)
        Q0_Q = 1 / E**2
        al3 = (alpha1 - alpha2 - Omegam0 - Omegar0) / 3
        f_Q = alpha1 - al3 * (Q0_Q)**2
        rootQ_correction = 0.5 * lambda0 * (Q0_Q)**0.5
        out = f_Q + rootQ_correction
        if np.any(out==0):
            return np.full_like(z, np.nan)

        return 1/out

    
    @classmethod
    def declare_parameters(cls):        # always use 'cls' for class methods
        return [
            Parameter(
                name="H0",
                latex=r"H_0",
                prior=UniformPrior(low=50.0, high=90.0),
                role="cosmo",
                status="free",
                proposed_scale=1.0
            ),
            Parameter(
                name="Omegam0",
                latex=r'\Omega_{m0}',
                prior=UniformPrior(low=0.0, high=1.0),
                role="cosmo",
                status="free",
                proposed_scale=0.015
            ),
            Parameter(
                name="alpha1",
                latex=r'\alpha_1',
                prior=UniformPrior(low=0.0, high=5.0),
                role="cosmo",
                status="fixed",
                value=1.0,
                proposed_scale=0.05
            ),
            Parameter(
                name="alpha2",
                latex=r'\alpha_2',
                prior=UniformPrior(low=0.0, high=1.0),
                role="cosmo",
                status="free", 
                proposed_scale=0.05
            ),
            Parameter(
                name="lambda0",
                latex=r'\lambda_0',
                prior=UniformPrior(low=-5.0, high=5.0),
                role="cosmo",
                status="free",
                value=0.0,
                proposed_scale=0.05
            )
        ]
 

@cosmix_registry.register_model("fQ_Hybrid_Curved")
class fQHybrid_Curved(CurvedfQBase):
    """
    Curved extension of fQHybrid.

    f(Q) = alpha1·Q + alpha2·Q₀ + alpha3·Q₀²/Q

    alpha3 is NOT a free parameter — it is derived from the Friedmann
    self-consistency condition at z=0 (P₀=0, B=0):

        3(α1−α3)H₀² + ½(α2+2α3)Q₀ + 3k(α1−α3) − 3H₀²(Ωm+Ωr) = 0

    Solving for α3:

        α3 = [3H₀²(Ωm+Ωr−α1) − 3k·α1 − ½α2·Q₀] / [Q₀ − 3(H₀²+k)]

    where k = −Ωk·H₀² and Q₀ is computed from _initial_conditions.

    In the flat limit (Ωk=0, Q₀=−6H₀²) this reduces exactly to
        α3 = (α1 − α2 − Ωm − Ωr) / 3
    matching the flat fQHybrid analytical constraint.

    Free parameters: H0, Omegam0, Omegak, gamma_0, alpha2
    Fixed parameter: alpha1 = 1 (same convention as flat class)
    Derived:         alpha3 (from Friedmann constraint above)
    """
    name = "fQ_Hybrid_Curved"

    def __init__(self, pm):
        super().__init__(pm)
        self._alpha1 = None
        self._alpha2 = None
        self._alpha3 = None
        self._Q0     = None

    def _set_params_from_theta(self, theta):
        H0      = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        Omegak  = self.pm.get_value(theta, "Omegak")
        gamma_0 = self.pm.get_value(theta, "gamma_0")
        self._alpha1 = self.pm.get_value(theta, "alpha1")
        self._alpha2 = self.pm.get_value(theta, "alpha2")

        # Q₀ from the non-metricity scalar at z=0 via the curved connection IC.
        try:
            y0 = self._initial_conditions(H0, Omegam0, Omegak, gamma_0)
            self._Q0 = y0[0]
        except (ValueError, RuntimeError):
            self._Q0 = -6.0 * H0**2   # flat-space fallback

        # α3 from Friedmann self-consistency at z=0 (P₀=0 → B=0).
        # Derived by substituting f, f' into _solve_H with H=H0, z=0, P=0
        # and requiring equality; see class docstring for the derivation.
        k_phys = -Omegak * H0**2
        denom  = self._Q0 - 3.0 * (H0**2 + k_phys)
        numer  = (3.0 * H0**2 * (Omegam0 + Omegar0 - self._alpha1)
                  - 3.0 * k_phys * self._alpha1
                  - 0.5 * self._alpha2 * self._Q0)
        self._alpha3 = numer / denom if denom != 0.0 else np.nan

    # ── Abstract interface ───────────────────────────────────────────────────
    def f(self, Q):
        return (self._alpha1 * Q
                + self._alpha2 * self._Q0
                + self._alpha3 * self._Q0**2 / Q)

    def f_prime(self, Q):
        return self._alpha1 - self._alpha3 * (self._Q0 / Q)**2

    def f_double_prime(self, Q):
        return 2.0 * self._alpha3 * self._Q0**2 / Q**3

    def f_triple_prime(self, Q):
        return -6.0 * self._alpha3 * self._Q0**2 / Q**4

    # ── Physicality guard ────────────────────────────────────────────────────
    def check_physicality(self, theta):
        self._set_params_from_theta(theta)
        # Q₀=0 makes f(Q) singular; non-finite α3 means the Friedmann
        # constraint denominator vanished (degenerate model point).
        if self._Q0 == 0.0 or not np.isfinite(self._alpha3):
            return False
        return super().check_physicality(theta)

    # ── Parameters ──────────────────────────────────────────────────────────
    @classmethod
    def declare_parameters(cls):
        return CurvedfQBase.declare_parameters() + [
            Parameter(
                name="alpha1",
                latex=r'\alpha_1',
                prior=UniformPrior(low=0.0, high=5.0),
                role="cosmo",
                status="fixed",
                value=1.0,
                proposed_scale=0.05
            ),
            Parameter(
                name="alpha2",
                latex=r'\alpha_2',
                prior=UniformPrior(low=0.0, high=1.0),
                role="cosmo",
                status="free",
                proposed_scale=0.05
            ),
        ]
