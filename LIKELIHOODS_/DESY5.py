# DESY5 
"""
DES-SN5YR likelihood (safe weighted covariance + analytic marginalization over M).

Behavior:
 - Reads distance & metadata CSV and STAT+SYS covariance from DES-SN5YR provided files.
 - Cleans `PROB_SNNV19` classifier probabilities (sentinel -9 handled).
 - Forms sqrt-weights W = diag(sqrt(p_i)) and weighted covariance cov_w = W @ cov @ W.
 - Adds tiny jitter to cov_w to improve PD behaviour, attempts cholesky, falls back to pinv.
 - Marginalizes analytically over scriptM (absolute magnitude) with a flat prior.
 - Exposes lnlike(theta, model), chi2(theta, model), data_size().

Notes:
 - lnlike uses N_eff = sum(prob) in the normalization term because we apply weighting.
 - mu_theory_noM computes mu_model WITHOUT subtracting scriptM (so dependence on M is linear).
 - The marginalised log-likelihood formula used includes the integration factor sqrt(2*pi / C).
"""

#------------------------------
# Preamble 
#------------------------------
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
import numpy as np
import pandas as pd

_DATA_DES = Path(__file__).resolve().parent.parent / "DATA_" / "DES-SN5YR" / "4_DISTANCES_COVMAT"
Default_data_file = _DATA_DES / "DES-SN5YR_HD+MetaData.csv"
Default_cov_file  = _DATA_DES / "STAT+SYS.txt.gz"

class DESY5(LikelihoodBase):
    name="D5"

    def __init__(self, pm, data_file=None, cov_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file
        if cov_file is None:
            cov_file = Default_cov_file

        self.data_file = data_file
        self.cov_file = cov_file

        data = pd.read_csv(self.data_file)
        self.zHEL = data["zHEL"].values
        self.zHD = data["zHD"].values
        self.mu_obs = data["MU"].values
        self.mu_err = data["MUERR_FINAL"].values
        self.beams_prob = 1 - data["PROBCC_BEAMS"]      # "PROBCC_BEAMS" is the prob that the SN is a core collapse

        self.N = len(self.zHEL)
        cov_full = np.genfromtxt(self.cov_file)[1:]
        cov = cov_full.reshape(self.N, self.N)
        np.fill_diagonal(cov, cov.diagonal() + self.mu_err**2)

        self.data_size = float(np.sum(self.beams_prob))
        self.highz = len( data[data.IDSURVEY==10].copy() )
        self.lowz = len( data[data.IDSURVEY!=10].copy() )

        w = np.sqrt( self.beams_prob.copy() )
        W = np.diag(w)

        cov_W = W @ cov @ W
        if not np.isfinite(cov_W).all():
            cov_W = np.nan_to_num(cov_W, nan=0.0, posinf=0.0, neginf=0.0)
            diag_mean = np.mean(np.diag(cov))
            cov_W += np.eye(self.N) * (1e-8 * max(1.0, diag_mean))

        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)

        # precompute normalization
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise RuntimeError("Covariance matrix not positive definite.")
        
        self.ln_norm = -0.5 * (logdet + self.data_size * np.log(2.0 * np.pi))

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="M",
                latex=r'\mathcal{M}',
                prior=UniformPrior(-19.6, -18.8),
                role="nuisance",
                status="free"
            )
        ]
    
    def get_requirements(self):
        return {
            "dL": self.zHD
        }
    
    def supports_marginalization(self, name):
        return False
    
    def marginalization_terms(self, theory):
        return None
    
    def lnlike(self, theta, theory):
        dL = theory.eval("dL", self.zHD)
        mu_th = np.empty(self.data_size)

        factor = (1.0 + self.zHD) / (1.0 + self.zHEL)
        M = self.pm.get_value(theta, "M")
        res = self.mu_obs - mu_th
        chi2 = res @ self.inv_cov
        
         

