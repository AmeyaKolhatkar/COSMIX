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
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from CORE_.BackgroundConfiguration import BackgroundConfig
from THEORY_.Solvers_.BackgroundProblem import AnalyticalProblem

from Constants import c, Omegar0

# ══════════════════════════════════════════════════════════════════════════════
# fQ_LCDM
# ══════════════════════════════════════════════════════════════════════════════
class fQ_LCDM(CosmologyModelBase):
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