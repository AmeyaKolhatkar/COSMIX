"""fQHybridIDE — f(Q) Hybrid model with Interacting Dark Energy (IDE).

Extends fQHybrid by allowing energy transfer between dark matter and dark
energy.  The matter density Ω_m(z) is no longer the standard (1+z)³
dilution but follows a coupled ODE:

    dΩ_m/dz = f(Ω_m, z; γ_m, γ_D)

The background H(z) is then solved numerically by a Numba-JIT RK4 solver.

Free parameters
---------------
H0       — Hubble constant today
Omegam0  — matter density today (initial condition for ODE)
alpha1   — f(Q) Hybrid amplitude
gamma_m  — dark-matter decay rate
gamma_D  — dark-energy injection rate
"""
import numpy as np
from numba import njit
from CORE_.CosmologyModelBase import CosmologyModelBase
from CORE_.ParameterManager_ import Parameter, UniformPrior
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
    alpha2 = params[1]
    gamma_m = params[2]
    gamma_D = params[3]
    
    Om0_tot = Omegam0 + Omegar0
    
    # 1. Compute Omega_r(z)
    Omr_z = Omegar0 * (1.0 + z)**4
    
    # 2. Compute the E^2(z) term
    Om_tilde = Om + Omr_z + alpha2
    Om_tilde_0 = Omegam0 + Omegar0 + alpha2 
    inner_sqrt = Om_tilde**2 + 2*(1-Om_tilde_0) 
    if inner_sqrt < 0.0:
        raise ValueError("Imaginary square root in E(z)")
    
    E2 = 0.5 * ( Om_tilde + np.sqrt(inner_sqrt) )  
    
    # 3. Compute dOm/dz
    numerator = 3.0 * Om * (1.0 + gamma_m) + 3 * gamma_D * ( alpha2 + E2 * (1 - Om_tilde_0) )    
    dOm_dz = numerator / (1.0 + z)
    
    return np.array([dOm_dz], dtype=np.float64)


class fQHybridIDE(CosmologyModelBase):
    """
    Interacting Dark Energy in f(Q) Gravity (Model I: Log-Square-Root)
    Integrates Omega_m(z) via Numba, then algebraically computes H(z).
    """
    name = "fQ_Hybrid_IDE"

    def check_physicality(self, theta):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha2 = self.pm.get_value(theta, "alpha2")

        if alpha2 <= 0.0:
            return False

        if Omegam0 + Omegar0 >= 1.0:
            return False 
        
        beta2 = (1 - Omegam0 - Omegar0 - alpha2) / 3
        fQ_0 = 1.0 + beta2
        
        # If fQ <= 0, G_eff is negative (repulsive gravity / ghost mode)
        if fQ_0 <= 1e-4:
            return False
        return True

    def background_problem(self, theta, z_grid):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        alpha2 = self.pm.get_value(theta, "alpha2")
        gamma_m = self.pm.get_value(theta, "gamma_m")
        gamma_D = self.pm.get_value(theta, "gamma_D")
        H0 = self.pm.get_value(theta, "H0")
        
        Om0 = Omegam0 + Omegar0

        params = np.array([Omegam0, alpha2, gamma_m, gamma_D], dtype=np.float64)
        y0 = np.array([Omegam0], dtype=np.float64)

        def extract(sol_y, z_grid):
            Omegam_grid = sol_y[0, :]
            Omegar_grid = Omegar0 * (1.0 + z_grid)**4
            Om_tilde_grid = Omegam_grid + Omegar_grid + alpha2
            inner_sqrt = Om_tilde_grid**2 + 4 * (1.0 - Om0 - alpha2)
            
            if np.any(inner_sqrt < 0):
                raise RuntimeError("Imaginary Square Root during H(z) evaluation")
                 
            H_grid = H0 * np.sqrt( 0.5 * ( Om_tilde_grid + np.sqrt(inner_sqrt) ) )
            
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
        alpha2 = self.pm.get_value(theta, "alpha2")
        lambda0 = self.pm.get_value(theta, "lambda0")

        E = bg_engine.E(z)
        beta2 = ( 1 - Omegam0 - Omegar0 - alpha2 ) / 3
        fQ = 1 - beta2 / E**2

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
                name="alpha2", 
                latex=r'\alpha_2', 
                prior=UniformPrior(low=0.0, high=np.inf), 
                role="cosmo", 
                status="free", 
                value=0.7,
                proposed_scale=0.05
            ),
            Parameter(
                name="gamma_m", 
                latex=r'\gamma_m', 
                prior=UniformPrior(low=-np.inf, high=np.inf), 
                role="cosmo", 
                status="free", 
                value=0.0,
                proposed_scale=0.05
                ),
            Parameter(
                name="gamma_D", 
                latex=r'\gamma_D', 
                prior=UniformPrior(low=-np.inf, high=np.inf), 
                role="cosmo", 
                status="free", 
                value=0.0,
                proposed_scale=0.05
                ),
            Parameter(
                name="lambda0", 
                latex=r'\lambda_0', 
                prior=UniformPrior(low=-np.inf, high=np.inf), 
                role="cosmo", 
                status="fixed", 
                value=0.0,
                proposed_scale=0.05
            )
        ]