"""
Modified RSD Likelihood for different secondary compilations. This one is specifically for implementing 
covariance matrix for WiggleZ survey.
"""
import numpy as np
import pandas as pd
from cosmix.core.Registry import cosmix_registry
from cosmix.core.Paths import DATA_DIR
from cosmix.likelihoods.RSD import RedshiftSpaceDistortion 

from pathlib import Path as _Path
Default_data_file = DATA_DIR / "RSD" / "RSD_WiggleZ.xlsx"
Default_cov_file = DATA_DIR / "RSD" / "WiggleZ_cov.txt"

@cosmix_registry.register_likelihood("rsdwigglez")
class RSDWiggleZ(RedshiftSpaceDistortion):
    name = "RSDWiggleZ"

    def __init__(self, pm, data_file=None, cov_file=None):
        if data_file is None:
            data_file = Default_data_file
        self.data_file = data_file 
        if cov_file is None:
            cov_file = Default_cov_file
        super().__init__(pm, data_file)
        self.cov_file = cov_file

        data = pd.read_excel(self.data_file)

        cov = np.loadtxt(self.cov_file) 
        assert cov.ndim == 2 and cov.shape[0] == cov.shape[1]
        self.inv_cov = np.linalg.inv(cov)
        self.data_size = len(self.fs8)

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise RuntimeError("Covariance matrix not positive definite.")
        
        self.ln_norm = -0.5 * (logdet + self.data_size * np.log(2.0 * np.pi))

    def lnlike(self, theta, theory):
        fs8_model = theory.eval("fsigma8", self.z)
        H_model = theory.eval("H", self.z)
        DM_model = theory.eval("DM", self.z)

        AP = (self.H_fid * self.DM_fid) / (H_model * DM_model)
        fs8_corrected = AP * fs8_model

        delta = self.fs8 - fs8_corrected
        chi2 = delta @ self.inv_cov @ delta
        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2 + self.ln_norm
    
    