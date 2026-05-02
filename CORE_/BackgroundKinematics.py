"""
Accepts H(z) and models parameters.

Computes cosmological quantities only once per theta based on H(z), like:
    - dL
    - DA

"chi" is the line-of-sight comoving distance or D_C given by
    c * int(dz'/H(z'), 0, z)

> All computations and integrals live here. 

> Cache keys must compute kinematics configuration
    e.g. cache_key = (theta, qty, z_grid_id, precision_id)

> The model provides a BackgroundProblem descriptor; this class calls .solve()
"""

import numpy as np
from CORE_.BackgroundConfiguration import BackgroundConfig
from CORE_.ObservableEngineBase import ObservableEngineBase
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid

c = 2.99792458e5            # km/s

class BackgroundKinematics(ObservableEngineBase):
    """
    Deterministic Background Kinematics Engine

        - Delegates background solve to the model's BackgroundProblem descriptor
        - Solves background equations sparsely
        - Integrates densely for precision
        - Provides fast derived distance measures via np.interp
        - Remains completely likelihood agnostic
    """
    capabilities = {
        "H", "H_cc_unc", "H_cc_cor", "dL", "mu", "DM", "DH", "DV", "muG"
    }

    def __init__(self, model, theta, config: BackgroundConfig):
        self.model = model
        self.theta = theta
        self.config = config

        self._z_grid = self._build_z_grid()         # Sparse solver grid
        assert np.all(np.diff(self._z_grid) > 0)
        self._z_dense = self._build_z_dense()       # Dense interpolation grid
        assert np.all(np.diff(self._z_dense) > 0)
        
        self._cache = {}                            # (per instance) evaluation cache
        self._solve_background()


    def _build_z_grid(self):
        """
        Construct a deterministic, non-uniform redshift grid.
        Dense at low-z for data precision, seamlessly extended to high-z for ODE solvers.
        """
        z_low = np.linspace(0.0, self.config.z_max, self.config.nz)

        if getattr(self.config, "z_max_extended", None) is not None:
            nz_ext = getattr(self.config, "nz_extended", 600)
            z_high = np.geomspace(self.config.z_max + 0.01, 1100.0, nz_ext)
            return np.concatenate((z_low, z_high))

        return z_low
    
    def _build_z_dense(self):
        """Dense grid that mirrors the sparse horizon for flawless np.interp."""
        nz_dense = getattr(self.config, "nz_dense", 600)
        z_low_dense = np.linspace(0.0, self.config.z_max, nz_dense)
        if getattr(self.config, "z_max_extended", None) is not None:
            nz_dense_ext = getattr(self.config, "nz_dense_extended", 600)
            z_high_dense = np.geomspace(self.config.z_max + 0.01, 1100.0, nz_dense_ext)
            return np.concatenate((z_low_dense, z_high_dense))
        return z_low_dense

    def _solve_background(self):
        """
        solve background sparsely, splines once, integrates densely, and caches for np.interp
        """
        # 1. Ask the model for a problem descriptor and solve it
        problem = self.model.background_problem(self.theta, self._z_grid)
        bk_dict = problem.solve(self._z_grid)
        H_sparse = bk_dict["H"]

        if "Omegam" in bk_dict:
            Om_sparse = bk_dict["Omegam"]
            Om_spline = CubicSpline(self._z_grid, Om_sparse)
            self.Om_dense = Om_spline(self._z_dense)
            self._has_custom_Om = True
        else:
            self._has_custom_Om = False
        
        # 2. Build a Cubic Spline once per MCMC step
        H_spline = CubicSpline(self._z_grid, H_sparse)
        
        # 3. Populate the dense arrays
        self._H_dense = H_spline(self._z_dense)
        self._H0 = self._H_dense[0]
        
        # 4. Integrate on the dense array for high accuracy
        inv_H = c / self._H_dense
        self._chi_dense = cumulative_trapezoid(inv_H, self._z_dense, initial=0.0)
        
        # 5. Compute derivatives on the dense array
        dH_dz_dense = H_spline(self._z_dense, 1)
        self._dlnH_dN_dense = -(1.0 + self._z_dense) * dH_dz_dense / self._H_dense


    # Preliminary Quantities
    def H(self, z):
        return np.interp(z, self._z_dense, self._H_dense)
    
    def chi(self, z):
        return np.interp(z, self._z_dense, self._chi_dense)

    def dlnH_dN(self, z):
        return np.interp(z, self._z_dense, self._dlnH_dN_dense)
    
    def H0(self):
        return self._H0
    
    def E(self, z):
        return self.H(z) / self.H0()
    
    # Distance measures    
    def DM(self, z):
        return self.chi(z)
    
    def DH(self, z):
        return c / self.H(z)
    
    def DV(self, z):
        return (z * self.DM(z)**2 * self.DH(z))**(1.0/3.0)

    def dL(self, z):
        return (1.0 + z) * self.chi(z)
    
    def mu(self, z):
        return 5.0 * np.log10(self.dL(z)) + 25.0
    
    def Omegamz(self, z):
        E = self.E(z)
        if self._has_custom_Om:
            omz_only = np.interp(z, self._z_dense, self.Om_dense)
        else:
            Om0 = self.model.pm.get_value(self.theta, "Omegam0") 
            omz_only = Om0 * (1+z)**3 

        return omz_only / E**2   
    
    def muG(self, z):
        return self.model.muG(z, self.theta, self)
    
    def z_grid(self):
        return self._z_grid.copy()