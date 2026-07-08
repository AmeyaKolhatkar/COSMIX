"""
LCDM implementation in f(Q) gravity; designed to test the root correction
f(Q) = alpha_1 Q + alpha_2 Q_0 + lambda_0 sqrt{QQ_0} 

Free parameters
---------------
H0       — Hubble constant today [km/s/Mpc]
Omegam0  — matter density parameter today
alpha1   — set to 1 for GR limit (fixed)
alpha2   — equal to 1 - Omegam0 - Omegar0 for flatness (fixed)
lambda0   — root correction amplitude 

Fixed constants
---------------
Omegar0  — radiation density (imported from Constants.py)
"""
import numpy as np
from CORE_.Registry import cosmix_registry
from CORE_.CosmologyModelBase import CosmologyModelBase
from THEORY_.CurvedfQBase import CurvedfQBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from CORE_.BackgroundConfiguration import BackgroundConfig
from THEORY_.Solvers_.BackgroundProblem import AnalyticalProblem

from Constants import c, Omegar0

# ══════════════════════════════════════════════════════════════════════════════
# fQ_LCDM
# ══════════════════════════════════════════════════════════════════════════════
@cosmix_registry.register_model("fQ_LCDM")
class fQLCDM(CosmologyModelBase):
    name = "fQ_LCDM"

    def __init__(self, pm):
        super().__init__(pm) 

    def _H(self, z, H0, Omegam0):
        z = np.asarray(z)
        OmegaL0 = 1 - Omegam0 - Omegar0 
        arg =  Omegam0 * (1.0 + z)**3 + Omegar0*(1+z)**4 + OmegaL0
        if np.any(arg <= 0):
            return np.full_like(z, np.nan)        
        return H0 * np.sqrt(arg)
    
    def background_problem(self, theta, z_grid):
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")

        return AnalyticalProblem(
            h_func=lambda z: self._H(z, H0, Omegam0)
        )
    
    def background_config(self):
        return BackgroundConfig(
            z_max=3.0,
            nz=150,
            integration_method="trapz"
        )
    
    def muG(self, z, theta, bg_engine):
        lambda0 = self.pm.get_value(theta, "lambda0")  

        E = bg_engine.E(z)
        rootQ_correction = 0.5 * lambda0 / E
        out = 1 + rootQ_correction
        if np.any(out==0):
            return np.full_like(z, np.nan)

        return 1/out 

    
    def build_engines(self, theta, requirements):
        """Override to replace CAMBKinematics with MGCAMBWrapper for fQ_LCDM.

        fQ_LCDM is background-inert: H(z) = ΛCDM, so the transfer function T(k)
        is identical to ΛCDM and only the growth factor D(z) is modified via μ_G(z).
        MGCAMBWrapper applies P_fQ(k,z) = P_ΛCDM(k,z) × [D_fQ(z)/D_fQ(0)]²,
        which is exact for this class of model and cheaper than running MGCAMB.
        """
        engines = super().build_engines(theta, requirements)

        needs_camb = any(
            name in ("Pk_linear", "Pk_nonlinear", "sigma8_camb", "r_drag")
            for name in requirements
        )
        if not needs_camb:
            return engines

        from CORE_.CAMBKinematics import MGCAMBWrapper
        from CORE_.GrowthKinematics import GrowthKinematics

        camb_engine   = next((e for e in engines if "Pk_linear" in e.capabilities), None)
        growth_engine = next((e for e in engines if "delta" in e.capabilities), None)

        if camb_engine is None or growth_engine is None:
            return engines  # fallback: something went wrong, leave as-is

        # Replace the plain CAMB engine with the growth-corrected wrapper
        engines = [e for e in engines if e is not camb_engine]
        engines.append(MGCAMBWrapper(camb_engine, growth_engine))

        return engines

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
                name="lambda0",
                latex=r'\lambda_0',
                prior=UniformPrior(low=-5.0, high=5.0),
                role="cosmo",
                status="free",
                value=0.0,
                proposed_scale=0.05
            )
        ]

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class fQLCDMCurved(CurvedfQBase):
    """
    f(Q) = alpha1 Q + alpha2 Q0  
    """
    name = "fQ_LCDM_Curved"

    def __init__(self, pm):
        super().__init__(pm)
        self.alpha1 = None
        self.alpha2 = None

    def _set_params_from_theta(self, theta):
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        Omegak = self.pm.get_value(theta, "Omegak")
        gamma0 = self.pm.get_value(theta, "gamma0")

        self.alpha1 = self.pm.get_value(theta, "alpha1")
        self.alpha2 = self.alpha1 - Omegam0 - Omegar0 - Omegak

        # ── Q₀ from kinematic IC (eq. 77 at z=0, γ̇₀=0) ─────────────────────
        try:
            y0 = self._initial_conditions(H0, Omegam0, Omegak, gamma0)
            self._Q0 = y0[0]
        except (ValueError, RuntimeError):
            self._Q0 = -6.0 * H0**2   # flat-space fallback

    # ── Abstract interface ───────────────────────────────────────────────────
    def f(self, Q):
        return self._alpha1 * Q + self._alpha2 * self._Q0

    def f_prime(self, Q):
        return self._alpha1 

    def f_double_prime(self, Q):
        return 0.0

    def f_triple_prime(self, Q):
        return 0.0
