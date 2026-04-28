"""
DES-Dovekie likelihood  (Vincenzi et al. 2024, DES-SN5YR cosmological analysis)

Data files (DATA_/DES-Dovekie/)
    DES-Dovekie_HD.csv       — 1820 distance moduli, redshifts, BEAMS probs
    covtot_inv_000.npz       — TOTAL (stat+sys) INVERSE covariance, stored as
                               the upper triangle of a 1820×1820 matrix in a
                               packed flat array (keys: 'nsn', 'cov')

The nuisance parameter M (absolute magnitude / H0 offset) is analytically
marginalized over a flat prior using the Conley et al. (2011) formula.
M and H0 are fully degenerate; do not use this sample alone to measure H0.
"""

#------------------------------
# Preamble 
#------------------------------
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
import numpy as np
import pandas as pd

_DATA_D5Dovekie = Path(__file__).resolve().parent.parent / "DATA_" / "DES-Dovekie"
Default_data_file = _DATA_D5Dovekie / "DES-Dovekie_HD.csv"
Default_cov_file  = _DATA_D5Dovekie / "covtot_inv_000.npz"

#------------------------------
# PPS Class Skeleton 
#------------------------------ 
class DESDovekie(LikelihoodBase):
    name="D5Dovekie"

    def __init__(self, pm, data_file=None, cov_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file
        if cov_file is None:
            cov_file = Default_cov_file

        self.data_file = data_file
        self.cov_file  = cov_file

        # ---- data vector ----
        data = pd.read_csv(self.data_file)
        mask = data["zHD"].values > 0.0      # keep all SNe with positive redshift
        self.mask = mask

        self.zCMB   = data["zHD"].values[mask]   # CMB-frame redshift (SNANA: zHD)
        self.zHEL   = data["zHEL"].values[mask]  # heliocentric redshift
        self.mu_obs = data["MU"].values[mask]     # observed distance modulus

        # ---- covariance ----
        # covtot_inv_000.npz stores the INVERSE covariance as the upper triangle
        # of the full N×N matrix (packed flat, keys: 'nsn', 'cov').
        d = np.load(self.cov_file)
        n = int(d["nsn"][0])
        inv_cov_full = np.zeros((n, n), dtype=float)
        inv_cov_full[np.triu_indices(n)] = d["cov"].astype(float)
        i_lower = np.tril_indices(n, -1)
        inv_cov_full[i_lower] = inv_cov_full.T[i_lower]   # symmetrize

        # Apply boolean mask → integer index array for np.ix_
        mask_idx       = np.where(mask)[0]
        self.inv_cov   = inv_cov_full[np.ix_(mask_idx, mask_idx)]

        # Full covariance (needed for per-point error bars in residual plots)
        self.cov = np.linalg.inv(self.inv_cov)

        # Normalization: logdet(cov) = −logdet(inv_cov), avoids double-inversion
        sign_inv, logdet_inv = np.linalg.slogdet(self.inv_cov)
        if sign_inv <= 0:
            raise RuntimeError("DES-Dovekie: inverse covariance not positive definite.")
        N = len(self.mu_obs)
        self.ln_norm = 0.5 * logdet_inv - 0.5 * N * np.log(2.0 * np.pi)

        self.N              = N
        self.ones           = np.ones(N)
        self.data_size      = N
        self.produce_residuals = True

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="M",
                latex=r'\mathcal{M}',
                prior= UniformPrior(-19.6, -18.8),
                role="nuisance",
                status="marginalized", 
                proposed_scale=0.001
            )
        ]
    
    def get_requirements(self):
        return {
            "dL": self.zCMB
        }
    
    def supports_marginalization(self, name):
        return name == "M"
    
    def marginalization_terms(self, theory):
        if not self.pm.is_marginalized("M"):
            return None
        dL = theory.eval("dL", self.zCMB)
        factor = (1.0 + self.zHEL) / (1.0 + self.zCMB)
        mu_th = 5.0 * np.log10(factor * dL) + 25.0
        res = self.mu_obs - mu_th
        A = float(self.ones @ self.inv_cov @ self.ones)
        B = float(self.ones @ self.inv_cov @ res)
        return GaussMargTerm("M", A=A, B=B, ln_norm=0.0)
    
    def lnlike(self, theta, theory):
        dL = theory.eval("dL", self.zCMB)

        # hubble flow
        zCMB = self.zCMB
        zHEL = self.zHEL
        assert np.all(zCMB>0)
        assert np.all(zHEL>0)
        factor = (1 + zHEL) / (1 + zCMB)
        mu_th = 5.0 * np.log10( factor*dL ) + 25.0

        if self.pm.is_marginalized("M"):
            res = self.mu_obs - mu_th
        else:
            M = self.pm.get_value(theta, "M")
            res = self.mu_obs - mu_th - M

        chi2 = res @ self.inv_cov @ res
        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2 + self.ln_norm
    
    def norm_term(self):
        return float(self.ln_norm)
    
    def _map_M(self, mu_th_noM):
        """Return MAP value of M when it is marginalized (B/A from Gaussian integral)."""
        r = self.mu_obs - mu_th_noM
        A = float(self.ones @ self.inv_cov @ self.ones)
        B = float(self.ones @ self.inv_cov @ r)
        return B / A

    def get_theory_components(self, theta, theory, z_override=None):
        dL_data = theory.eval("dL", self.zCMB)
        factor = (1 + self.zHEL) / (1 + self.zCMB)
        mu_th_noM = 5.0 * np.log10(factor * dL_data) + 25.0

        if self.pm.is_marginalized("M"):
            M = self._map_M(mu_th_noM)
        else:
            M = self.pm.get_value(theta, "M")

        if z_override is None:
            x = self.zCMB
            d_vec = self.mu_obs
            th_vec = mu_th_noM + M
            sigma = np.sqrt(np.diag(self.cov))
            return {"mu": (x, d_vec, th_vec, sigma)}
        else:
            x = z_override
            dL = theory.eval("dL", x)
            mu_th = 5.0 * np.log10(dL) + 25.0
            return {"mu": (x, None, mu_th + M, None)}
    
    def get_plots(self):
        return {
            "mu": True
        }