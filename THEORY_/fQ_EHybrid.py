"""fQEHybrid — Extended f(Q) Hybrid modified gravity model.

Extends the Hybrid model by replacing the analytical Friedmann solution
with a Numba-JIT ODE integration (RK4).  The modified Friedmann equation
is solved numerically from z=0 upward, enabling more general f(Q) forms
that do not have analytical inverses.

Free parameters
---------------
H0       — Hubble constant today
Omegam0  — matter density today
alpha1   — coefficient of the modified gravity term
alpha2   — cosmological constant analogue (Ω_Λ-like term)
n        — power-law index of the f(Q) deviation
"""
import numpy as np
from numba import njit
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from CORE_.BackgroundConfiguration import BackgroundConfig
from THEORY_.Solvers_.BackgroundProblem import NumbaODEProblem

from Constants import c, Omegar0

@njit
def fQ_rhs(z, y, params):
    u = y[0]
    Omegam0 = params[0]
    alpha1 = params[1]
    alpha2 = params[2]
    n = params[3]
    Omegar0 = params[4]
    
    if u <= 0.0:
        raise ValueError("[fQ_EHybrid] Imaginary H")

    Omega_tilde = Omegam0 * (1.0 + z)**3 + Omegar0 * (1.0 + z)**4 + alpha2
    Omega_tilde_prime = 3.0 * Omegam0 * (1.0 + z)**2 + 4.0 * Omegar0 * (1.0 + z)**3

    denominator = alpha1 * (n + 1.0) * u - n * Omega_tilde
    
    if abs(denominator) < 1e-6:
        raise ValueError("[fQ_EHybrid] Phantom Crossing")

    du_dz = (Omega_tilde_prime * u) / denominator

    if du_dz < 0.0:
        raise ValueError("[fQ_EHybrid] Negative Derivative")

    return np.array([du_dz], dtype=np.float64)

class fQEHybrid(CosmologyModelBase):
    """
    f(Q) = alpha1 Q + alpha2 Q0 + alpha3 Q0 (Q0/Q)^n

    H(z) obtained by solving the differential equation
    u' = Omega_tilde' u / ( alpha1 u (n+1) - n Omega_tilde )

    present day condition: (2n+1)alpha3 = alpha1 - Omega_tilde_0
    """
    name="fQ_EHybrid"

    def __init__(self, pm):
        super().__init__(pm)
    
    def background_problem(self, theta, z_grid):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha1 = self.pm.get_value(theta, "alpha1")
        alpha2 = self.pm.get_value(theta, "alpha2")
        n = self.pm.get_value(theta, "n")
        H0 = self.pm.get_value(theta, "H0")

        params = np.array([Omegam0, alpha1, alpha2, n, Omegar0], dtype=np.float64)
        y0 = np.array([1.0], dtype=np.float64)

        def extract(sol_y, z_grid):
            u_array = sol_y[0, :]
            return {"H": H0 * np.sqrt(u_array)}

        return NumbaODEProblem(
            rhs=fQ_rhs,
            y0=y0,
            params=params,
            extract=extract
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
        n = self.pm.get_value(theta, "n")
        lambda0 = self.pm.get_value(theta, "lambda0")

        E = bg_engine.E(z)
        Q0_Q = 1 / E**2
        al3 = (alpha1 - alpha2 - Omegam0 - Omegar0) / (2*n+1)
        f_Q = alpha1 - n * al3 * (Q0_Q)**(n+1)
        rootQ_correction = 0.5 * lambda0 * (Q0_Q)**0.5
        out = f_Q + rootQ_correction
        if np.any(out==0):
            return np.full_like(z, np.nan)

        return 1/out
    
    def check_physicality(self, theta):
        """
        Fails fast if the present-day universe contains a mathematical singularity
        or negative expansion derivative.
        """
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha1 = self.pm.get_value(theta, "alpha1")
        alpha2 = self.pm.get_value(theta, "alpha2")
        n = self.pm.get_value(theta, "n")

        Omega_tilde_0 = Omegam0 + Omegar0 + alpha2
        
        # The ODE denominator at z=0 (since u = 1)
        denominator_0 = alpha1 * (n + 1) - n * Omega_tilde_0
        
        # If the denominator is zero/negative, u' is infinite/negative today.
        if denominator_0 <= 1e-5:
            #print(f"[fQ_EHybrid] REJECTED at z=0: Denominator = {denominator_0:.4f} | alpha1={alpha1}, alpha2={alpha2}, n={n}")
            return False
            
        return True
    
    @classmethod
    def declare_parameters(cls):        # always use 'cls' for class methods
        return [
            Parameter(
                name="H0",
                latex=r"H_0",
                prior=UniformPrior(low=50.0, high=90.0),
                role="cosmo",
                status="free",
                value=70.0,
                proposed_scale=1.0
            ),
            Parameter(
                name="Omegam0",
                latex=r'\Omega_{m0}',
                prior=UniformPrior(low=0.0, high=1.0),
                role="cosmo",
                status="free",
                value=0.3,
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
                value=0.7,
                proposed_scale=0.05
            ),
            Parameter(
                name="n",
                latex=r'n',
                prior=UniformPrior(low=0.0, high=10.0),
                role="cosmo",
                status="free",
                value=1.0,
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