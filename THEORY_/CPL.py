"""Implements the CPL/ w0-wa CDM cosmology with the EoS:

    w(z) = w0 + wa * z / (1+z)
"""
import numpy as np
from CORE_.Registry import cosmix_registry
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from CORE_.BackgroundConfiguration import BackgroundConfig
from THEORY_.Solvers_.BackgroundProblem import AnalyticalProblem 

from Constants import c, Omegar0

# ══════════════════════════════════════════════════════════════════════════════
# w0waCDM
# ══════════════════════════════════════════════════════════════════════════════
@cosmix_registry.register_model("CPL")
class CPL(CosmologyModelBase):
    name="CPL"

    def __init__(self, pm):
        super().__init__(pm)

    def _H(self, z, H0, Omegam0, Omegar0, w0, wa):
        z = np.asarray(z)
        omegaDE = (1.0 + z)**(3.0 * ( 1.0 + w0 + wa )) * np.exp(3.0*wa/(1+z))
        arg = Omegam0 * (1.0 + z)**3 + Omegar0*(1.0 + z)**4 + (1.0 - Omegam0 - Omegar0) * omegaDE
        if np.any(arg <= 0):
            return np.full_like(z, np.nan)
        return H0 * np.sqrt(arg)
    
    def background_problem(self, theta, z_grid):
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        w0 = self.pm.get_value(theta, "w0")
        wa = self.pm.get_value(theta, "wa")

        return AnalyticalProblem(
            h_func=lambda z: self._H(z, H0, Omegam0, Omegar0, w0, wa)
        )
    
    def background_config(self):
        return BackgroundConfig(
            z_max=3.0,
            nz=150,
            integration_method="trapz"
        )
    
    @classmethod
    def declare_parameters(cls):
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
                name="w0",
                latex=r"w_0",
                prior=UniformPrior(low=-3.0, high=1.0),      # DESI DR2
                role="cosmo",
                status="free",
                proposed_scale=0.05
            ),
            Parameter(
                name="wa",
                latex=r"w_a",
                prior=UniformPrior(low=-3.0, high=2.0),      # DESI DR2
                role="cosmo",
                status="free",
                proposed_scale=0.05
            )
        ]