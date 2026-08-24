"""
mu0, Sigma0 parametrization of Modified Gravity
"""
import numpy as np
from cosmix.core.Registry import cosmix_registry
from cosmix.core.CosmologyModelBase import CosmologyModelBase
from cosmix.core.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from cosmix.core.BackgroundConfiguration import BackgroundConfig
from cosmix.theory.Solvers_.BackgroundProblem import AnalyticalProblem

from cosmix.Constants import c, Omegar0

@cosmix_registry.register_model("MG_muSig")
class muSigma(CosmologyModelBase):
    name = "MG_muSig"

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
        """
        Part of the mu-Sigma parametrization -
        mu(z) = 1 + mu0 OmegaDE(z)/OmegaDE0 
        Sigma(z) = 1 + Sigma0 OmegaDE(z)/OmegaDE0 

        mu(z) = 1 + mu0 * OmegaDE(z)/OmegaDE0,  and for Lambda
        OmegaDE(z)/OmegaDE0 = 1/E^2(z)  since rho_DE is constant.
        """
        mu0 = self.pm.get_value(theta, "mu0")
        
        return 1.0 + mu0 / bg_engine.E(z)**2

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="H0",
                latex=r"H_0",
                prior=UniformPrior(low=40.0, high=100.0),
                role="cosmo",
                status="free",
                value=67.4,
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
                name="mu0",
                latex=r'\mu_0',
                prior=UniformPrior(low=-2.0, high=2.0),
                role="cosmo",
                status="free",
                value=0.0,
                proposed_scale=0.05
            )
        ]
