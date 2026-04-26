"""fQLSRIDE — f(Q) Log-Square-Root model with Interacting Dark Energy (IDE).

Couples the LSR f(Q) background to an IDE sector where dark matter and
dark energy exchange energy at rates γ_m and γ_D.  The Ω_m(z) evolution
is integrated by a Numba-JIT ODE solver; H(z) is then obtained from the
modified Friedmann equation.

Free parameters
---------------
H0       — Hubble constant today
Omegam0  — matter density today
alpha1   — LSR amplitude
beta     — LSR scale factor
gamma_m  — dark-matter decay rate
gamma_D  — dark-energy injection rate
"""
import numpy as np
from numba import njit
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
from CORE_.BackgroundConfiguration import BackgroundConfig
from THEORY_.Solvers_.BackgroundProblem import NumbaODEProblem
from Constants import Omegar0

@njit
def ide_Omegam_rhs(z, y, params):
    """
    Integrates d(Omega_m)/dz.
    y[0] is Omega_m(z).
    """
    Om = y[0]
    Omegam0 = params[0]
    gamma_m = params[1]
    gamma_D = params[2]
    
    Om0_tot = Omegam0 + Omegar0
    
    # 1. Compute Omega_r(z)
    Omr_z = Omegar0 * (1.0 + z)**4
    
    # 2. Compute the A(z) term
    inner_sqrt = 1.0 + (4.0 * (Om + Omr_z)) / ((1.0 - Om0_tot)**2)
    if inner_sqrt < 0.0:
        raise ValueError("Imaginary Square Root in A(z)")
    
    A = 1.0 + np.sqrt(inner_sqrt)
    
    # 3. Compute dOm/dz
    numerator = 3.0 * Om * (1.0 + gamma_m) + 1.5 * gamma_D * ((1.0 - Om0_tot)**2) * A
    dOm_dz = numerator / (1.0 + z)
    
    return np.array([dOm_dz], dtype=np.float64)


class fQLSRIDE(CosmologyModelBase):
    """
    Interacting Dark Energy in f(Q) Gravity (Model I: Log-Square-Root)
    Integrates Omega_m(z) via Numba, then algebraically computes H(z).
    """
    name = "fQ_LSR_IDE"

    def check_physicality(self, theta):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        beta = self.pm.get_value(theta, "beta")

        if beta <= 0.0:
            return False

        if Omegam0 + Omegar0 >= 1.0:
            return False # Avoids division by zero in A(z)
        
        term1_0 = 0.5 * np.log(beta) - 1.0
        term2_0 = 0.5 * (1.0 - Omegam0 - Omegar0) * term1_0
        fQ_0 = 1.0 + term2_0
        
        # If fQ <= 0, G_eff is negative (repulsive gravity / ghost mode)
        if fQ_0 <= 1e-4:
            return False
        return True

    def background_problem(self, theta, z_grid):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        gamma_m = self.pm.get_value(theta, "gamma_m")
        gamma_D = self.pm.get_value(theta, "gamma_D")
        H0 = self.pm.get_value(theta, "H0")
        
        Om0 = Omegam0 + Omegar0

        params = np.array([Omegam0, gamma_m, gamma_D], dtype=np.float64)
        y0 = np.array([Omegam0], dtype=np.float64)

        def extract(sol_y, z_grid):
            Omegam_grid = sol_y[0, :]
            Omegar_grid = Omegar0 * (1.0 + z_grid)**4
            inner_sqrt = 1.0 + (4.0 * (Omegam_grid + Omegar_grid)) / ((1.0 - Om0)**2)
            
            if np.any(inner_sqrt < 0):
                raise RuntimeError("Imaginary Square Root during H(z) evaluation")
                 
            A_grid = 1.0 + np.sqrt(inner_sqrt)
            H_grid = 0.5 * H0 * (1.0 - Om0) * A_grid
            
            return {"H": H_grid, "Omegam": Omegam_grid}

        return NumbaODEProblem(
            rhs=ide_Omegam_rhs,
            y0=y0,
            params=params,
            extract=extract
        )

    def background_config(self):
        return BackgroundConfig(z_max=3.0, nz=500, integration_method="trapz")

    def muG(self, z, theta, bg_engine):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        beta = self.pm.get_value(theta, "beta")
        lambda0 = self.pm.get_value(theta, "lambda0")

        E = bg_engine.E(z)

        term1 = 0.5 * np.log(beta/E**2) - 1
        term2 = 0.5 * (1 - Omegam0 - Omegar0) * term1 / E
        fQ = 1 + term2

        rootQ_correction = 0.5 * lambda0 / E
        out = fQ + rootQ_correction

        if np.any(fQ<=0):
            return np.full_like(z, np.nan)

        return 1/out
    
    
    @classmethod
    def declare_parameters(cls):
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
                name="beta", 
                latex=r'\beta', 
                prior=UniformPrior(low=0.0, high=10.0), 
                role="cosmo", 
                status="fixed", 
                value=1.0,
                proposed_scale=0.05
            ),
            Parameter(
                name="gamma_m", 
                latex=r'\gamma_m', 
                prior=UniformPrior(low=-5.0, high=5.0), 
                role="cosmo", 
                status="free", 
                value=0.0,
                proposed_scale=0.05
                ),
            Parameter(
                name="gamma_D", 
                latex=r'\gamma_D', 
                prior=UniformPrior(low=-5.0, high=5.0), 
                role="cosmo", 
                status="free", 
                value=0.0,
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