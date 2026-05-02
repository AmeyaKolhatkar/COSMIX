"""PantheonplusSH0ES — Pantheon+ likelihood anchored with the SH0ES H₀ prior.

Identical to Pantheonplus but uses the full SH0ES-calibrated data vector,
which includes additional Cepheid-calibrated low-z supernovae.  This makes
the likelihood informative about H₀ (unlike the uncalibrated Pantheonplus).

Use PantheonplusSH0ES when you want the SN data to constrain H₀ directly.
Use Pantheonplus (+ optional SH0ESprior) for more flexibility.

Data files: same as Pantheonplus (DATA_/Pantheon/).
"""
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
import numpy as np
import pandas as pd

_DATA_PP = Path(__file__).resolve().parent.parent / "DATA_" / "Pantheon"
Default_data_file = _DATA_PP / "Pantheon+SH0ES.dat"
Default_cov_file  = _DATA_PP / "Pantheon+SH0ES_STAT+SYS.cov"


# ══════════════════════════════════════════════════════════════════════════════
class PantheonplusSH0ES(LikelihoodBase):
    name="PPS"

    def __init__(self, pm, data_file=None, cov_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file
        if cov_file is None:
            cov_file = Default_cov_file

        self.data_file = data_file
        self.cov_file = cov_file

        # construction from data
        data = pd.read_csv(self.data_file, sep=r"\s+")
        self.orig_len = len(data)
        self.mask = ( (data["zHD"] > 0.01) | (data["IS_CALIBRATOR"].astype(bool)) )
        self.zCMB = data["zHD"][self.mask].values
        self.zHEL = data["zHEL"][self.mask].values
        self.m_obs = data["m_b_corr"][self.mask].values
        self.is_cal = data["IS_CALIBRATOR"][self.mask].astype(bool).values
        self.mu_ceph = data["CEPH_DIST"][self.mask].values

        cov_full = np.loadtxt(self.cov_file, skiprows=1)
        cov_full = cov_full.reshape(self.orig_len, self.orig_len)
        cov = cov_full[np.ix_(self.mask, self.mask)]

        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)

        # precompute normalization
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise RuntimeError("Covariance matrix not positive definite")
        
        self.ln_norm = -0.5 * (logdet + len(self.m_obs) * np.log(2.0 * np.pi))

        self.N = len(self.m_obs)
        self.ones = np.ones(self.N)
        self.data_size = self.N
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
                proposed_scale=0.01
            )
        ]
    
    def get_requirements(self):
        return {
            "dL": self.zCMB[~self.is_cal]
        }
    
    def supports_marginalization(self, name):
        return name == "M"
    
    def marginalization_terms(self, theory):
        if not self.pm.is_marginalized("M"):
            return None
        dL = theory.eval("dL", self.zCMB[~self.is_cal])
        mu_th = np.empty(self.N)
        mu_th[self.is_cal] = self.mu_ceph[self.is_cal]
        zCMB = self.zCMB[~self.is_cal]
        zHEL = self.zHEL[~self.is_cal]
        factor = (1 + zHEL) / (1 + zCMB)
        mu_th[~self.is_cal] = 5.0 * np.log10(factor * dL) + 25.0
        r = self.m_obs - mu_th
        A = float(self.ones @ self.inv_cov @ self.ones)
        B = float(self.ones @ self.inv_cov @ r)
        return GaussMargTerm("M", A=A, B=B, ln_norm=0.0)
    
    def lnlike(self, theta, theory):
        dL = theory.eval("dL", self.zCMB[~self.is_cal])
        mu_th = np.empty(self.N)

        # cepheids
        mu_th[self.is_cal] = self.mu_ceph[self.is_cal]
        # hubble flow
        zCMB = self.zCMB[~self.is_cal]
        zHEL = self.zHEL[~self.is_cal]
        assert np.all(zCMB>0)
        assert np.all(zHEL>0)
        factor = (1 + zHEL) / (1 + zCMB)
        mu_th[~self.is_cal] = 5.0 * np.log10( factor*dL ) + 25.0
        # add absolute magnitude
        if self.pm.is_marginalized("M"):
            res = self.m_obs - mu_th
        else:
            M = self.pm.get_value(theta, "M")
            res = self.m_obs - mu_th - M
        # Gaussian log likelihood
        chi2 = res @ self.inv_cov @ res
        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2 + self.ln_norm
    
    def norm_term(self):
        return float(self.ln_norm)
    
    def _map_M(self, mu_th_noM):
        """Return MAP value of M when it is marginalized (B/A from Gaussian integral)."""
        r = self.m_obs - mu_th_noM
        A = float(self.ones @ self.inv_cov @ self.ones)
        B = float(self.ones @ self.inv_cov @ r)
        return B / A

    def get_theory_components(self, theta, theory, z_override=None):
        dL = theory.eval("dL", self.zCMB[~self.is_cal])
        mu_th = np.empty(self.N)
        mu_th[self.is_cal] = self.mu_ceph[self.is_cal]
        zCMB = self.zCMB[~self.is_cal]
        zHEL = self.zHEL[~self.is_cal]
        factor = (1 + zHEL) / (1 + zCMB)
        mu_th[~self.is_cal] = 5.0 * np.log10(factor * dL) + 25.0

        if self.pm.is_marginalized("M"):
            M = self._map_M(mu_th)
        else:
            M = self.pm.get_value(theta, "M")

        if z_override is None:
            x = self.zCMB
            d_vec = self.m_obs
            th_vec = mu_th + M
            sigma = np.sqrt(np.diag(self.cov))
            return {"mu": (x, d_vec, th_vec, sigma)}
        else:
            x = np.asarray(z_override)
            dL_ov = theory.eval("dL", x)
            mu_ov = 5.0 * np.log10(dL_ov) + 25.0
            return {"mu": (x, None, mu_ov + M, None)}
    
    def get_plots(self):
        return {
            "mu": True
        }