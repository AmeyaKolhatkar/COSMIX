"""fQLSR — f(Q) Log-Square-Root modified gravity model.

Implements  f(Q) = Q + alpha1 Q₀ sqrt(Q / (alpha2 Q₀)) ln(alpha2 Q₀ / Q).

The Hubble rate is obtained analytically:

    H(z) = H₀ ( 0.5(1-Ω_m0-Ω_r0) + sqrt(0.25(1-Ω_m0-Ω_r0)² + Ω(z)) )¹²

Free parameters
---------------
H0       — Hubble constant today
Omegam0  — matter density today
alpha1   — amplitude of the log-square-root correction
"""
import numpy as np
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from CORE_.BackgroundConfiguration import BackgroundConfig
from THEORY_.Solvers_.BackgroundProblem import AnalyticalProblem

from Constants import c, Omegar0

class fQLSR(CosmologyModelBase):
    """
    f(Q) = Q + alpha1 Q0 sqrt(Q / alpha2 Q0) ln ( alpha2 Q0 / Q )

    H(z) = H0 ( 0.5(1-Omegam0-Omegar0) + sqrt( 0.25(1-Omegam0-Omegar0)^2 + Omega(z) ) )
    """
    name="fQ_LSR"

    def __init__(self, pm):
        super().__init__(pm)

    def _H(self, z, H0, Omegam0, alpha1):
        z=np.asarray(z)
        omegaz = Omegam0*(1+z)**3 + Omegar0*(1+z)**4
        omegaz0 = Omegam0 + Omegar0
        omega_tilde_0 = 0.5 * (1 - omegaz0)
        sqrt_arg = omega_tilde_0**2 + omegaz
        if np.any(sqrt_arg < 0):
            return np.full_like(z, np.nan)
        arg = omega_tilde_0 + np.sqrt(sqrt_arg)
        if np.any(arg < 0):
            return np.full_like(z, np.nan)
        
        return H0 * arg
    
    def background_problem(self, theta, z_grid):
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha1 = self.pm.get_value(theta, "alpha1")

        return AnalyticalProblem(
            h_func=lambda z: self._H(z, H0, Omegam0, alpha1)
        )
    
    def background_config(self):
        return BackgroundConfig(
            z_max=3.0,
            nz=500,
            integration_method="trapz"
        )

    def muG(self, z, theta, bg_engine):
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha1 = self.pm.get_value(theta, "alpha1")

        H = bg_engine.H(z) 
        Q = 6*H**2
        Q0 = 6*H0**2
        al2 = (2 * alpha1 / (1 - Omegam0 - Omegar0))**2
        f_Q = 1 + alpha1 * (Q0 / (al2 * Q))**0.5 * ( 0.5 * np.log(al2 * Q0 / Q) - 1 )

        return 1/f_Q
    
        
    @classmethod
    def declare_parameters(cls):        # always use 'cls' for class methods
        return [
            Parameter(
                name="H0",
                latex=r"H_0",
                prior=GaussianPrior(low=50.0, high=90.0, mean=70.0, sig=5.0),
                role="cosmo",
                status="free"
            ),
            Parameter(
                name="Omegam0",
                latex=r'\Omega_{m0}',
                prior=GaussianPrior(low=0.1, high=0.5, mean=0.3, sig=0.05),
                role="cosmo",
                status="free"
            ),
            Parameter(
                name="alpha1",
                latex=r'\alpha_1',
                prior=UniformPrior(low=0.0, high=5.0),
                role="cosmo",
                status="free"
            )
        ]