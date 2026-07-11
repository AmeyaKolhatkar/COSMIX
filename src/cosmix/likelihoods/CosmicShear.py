"""
!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!
"""

from pathlib import Path
from cosmix.core.LikelihoodBase_ import LikelihoodBase
from cosmix.core.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
import numpy as np
import pandas as pd

class CosmicShearLikelihood(LikelihoodBase):
    name="Shear"

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="A_IA",
                latex=r"$A_\mathrm{IA}$",
                prior=UniformPrior(-5.0, 5.0),
                role="nuisance",
                status="free",
                value=0.0
            ),
            Parameter(
                name="eta_IA",
                latex=r"\eta_\mathrm{IA}",
                prior=UniformPrior(-5.0, 5.0),
                role="nuisance",
                status="fixed",
                value=0.0
            )
        ]
    
    def get_requirements(self):
        return {
            "Pk_linear": {"k": self._k_grid, "z": self._z_grid},
            "chi": self._z_grid,
            "H": self._z_grid
        }
    
    def lnlike(self, theta, theory):
        Pk = theory.eval_grid("Pk_linear", self._k_grid, self._z_grid)
        chi = theory.eval("chi", self._z_grid)
        H = theory.eval("H", self._z_grid)

        # 1. For fQ_LCDM: Sigma(z) = muG(z) = exact result under QSA
        #    (P_fQ already encodes this via MGCAMBWrapper; no separate Sigma factor needed)

        # 2. Apply photo-z shifts to n_i(chi)
        # 3. Recompute lensing kernels q_i(chi)
        # 4. Limber integral → C_ell^ij theory vector
        # 5. Add NLA intrinsic alignment contributions
        # 6. Apply multiplicative bias: C_ell *= (1+m_i)(1+m_j)
        # 7. Gaussian likelihood: -0.5 * dC @ C_inv @ dC + log_det_term