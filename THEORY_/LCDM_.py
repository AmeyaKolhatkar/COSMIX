"""LCDM — standard Lambda Cold Dark Matter cosmological model.

Implements the flat LCDM Hubble rate analytically:

    H(z) = H₀ sqrt( Ω_m0 (1+z)³ + Ω_r0 (1+z)⁴ + Ω_Λ )

where Ω_Λ = 1 - Ω_m0 - Ω_r0 (flatness constraint).

Free parameters
---------------
H0       — Hubble constant today [km/s/Mpc]
Omegam0  — matter density parameter today

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

r = Omegar0  # kept for backward compatibility within this module

# ══════════════════════════════════════════════════════════════════════════════
# LCDM
# ══════════════════════════════════════════════════════════════════════════════
class LCDM(CosmologyModelBase):
    name = "LCDM"

    def __init__(self, pm):
        super().__init__(pm)

    def _H(self, z, H0, Omegam0, Omegak0):
        z = np.asarray(z)
        OmegaL = 1 - Omegam0 - r - Omegak0 
        arg =  Omegam0 * (1.0 + z)**3 + r*(1+z)**4 + Omegak0*(1+z)**2 + OmegaL
        if np.any(arg <= 0):
            return np.full_like(z, np.nan)        
        return H0 * np.sqrt(arg)
    
    def background_problem(self, theta, z_grid):
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        Omegak0 = self.pm.get_value(theta, "Omegak0")

        return AnalyticalProblem(
            h_func=lambda z: self._H(z, H0, Omegam0, Omegak0)
        )
    
    def background_config(self):
        return BackgroundConfig(
            z_max=3.0,
            nz=150,
            integration_method="trapz"
        )

    
    @classmethod
    def declare_parameters(cls):        # always use 'cls' for class methods
        return [
            Parameter(
                name="H0",
                latex=r"H_0",
                prior=UniformPrior(low=50.0, high=90.0),
                role="cosmo",
                status="free",
                proposed_scale=0.1
            ),
            Parameter(
                name="Omegam0",
                latex=r'\Omega_{m0}',
                prior=UniformPrior(low=0.0, high=1.0),
                role="cosmo",
                status="free",
                proposed_scale=0.005
            ),
            Parameter(
                name="Omegak0",
                latex=r'\Omega_{k0}',
                prior=UniformPrior(low=-0.2, high=0.5),
                role="cosmo",
                status="fixed",
                value=0.0,
                proposed_scale=0.01
            )
        ]