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
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from CORE_.BackgroundConfiguration import BackgroundConfig
from THEORY_.Solvers_.BackgroundProblem import AnalyticalProblem

from Constants import c, Omegar0

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
            nz=500,
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
                status="fixed",
                value=0.0,
                proposed_scale=0.05
            )
        ]
 