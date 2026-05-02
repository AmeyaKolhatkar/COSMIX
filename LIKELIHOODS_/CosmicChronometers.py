"""CosmicChronometers — H(z) likelihood from differential-age dating.

Combines two complementary CC datasets:

1. Uncorrelated CC measurements
   Source: CC_NEW/CC_Uncorrelated.xlsx
   Uses diagonal errors; modelled as independent Gaussians.

2. Correlated CC measurements (Moresco et al. 2020)
   Source: CC_NEW/CCcovariance/data/
   Includes systematic covariance from stellar population models:
   initial mass function (IMF), spectral library (SLIB), stellar
   population synthesis (SPS), and further systematic uncertainties.
   The full covariance matrix is C_stat + sum(C_sys).

Data files live under DATA_/CC_NEW/ relative to the repository root.
"""
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, GaussianPrior
import numpy as np
import pandas as pd

_DATA_CC = Path(__file__).resolve().parent.parent / "DATA_" / "CC_NEW"
Default_data_unc_file = _DATA_CC / "CC_Uncorrelated.xlsx"
Default_data_cor_file = _DATA_CC / "CCcovariance" / "data" / "data_MM20.dat"
Default_Hz_cor_file   = _DATA_CC / "CCcovariance" / "data" / "HzTable_MM_BC03.dat"


# ══════════════════════════════════════════════════════════════════════════════
# CosmicChronometers
# ══════════════════════════════════════════════════════════════════════════════
class CosmicChronometers(LikelihoodBase):
    name="CC"

    def __init__(self, pm, unc_data_file=None, cor_data_file=None, cor_Hz_file=None):
        super().__init__(pm)
        if unc_data_file is None:
            unc_data_file = Default_data_unc_file
        if cor_data_file is None:
            cor_data_file = Default_data_cor_file
        if cor_Hz_file is None:
            cor_Hz_file = Default_Hz_cor_file

        # Uncorrelated data
        self.data_unc = pd.read_excel(unc_data_file)
        self.z_unc = self.data_unc["z"].values
        self.H_unc = self.data_unc["Hz"].values
        self.H_err_unc = self.data_unc["Hz_err"].values
        self.N_unc = len(self.z_unc)

        # Correlated data
        self.zmod, self.imf, self.slib, self.sps, self.spsooo = np.genfromtxt( cor_data_file, comments='#', usecols=(0,1,2,3,4), unpack=True) 
        self.z_cor, self.H_cor, self.H_err_cor, self.H_stat_err_cor, self.H_met_err_cor = np.genfromtxt(cor_Hz_file, comments='#', usecols=(0,1,2,3,4), unpack=True, delimiter=',')
        self.N_cor = len(self.z_cor)
        # COV MATS
        cov_diag = np.diag(self.H_err_cor**2)
        cov_stat = np.diag(self.H_stat_err_cor**2)
        cov_met = np.diag(self.H_met_err_cor**2)

        imf_intp = np.interp(self.z_cor, self.zmod, self.imf) / 100
        slib_intp = np.interp(self.z_cor, self.zmod, self.slib) / 100
        sps_intp = np.interp(self.z_cor, self.zmod, self.sps) / 100
        spsooo_intp = np.interp(self.z_cor, self.zmod, self.spsooo) / 100

        cov_imf = np.outer(self.H_cor * imf_intp, self.H_cor * imf_intp)
        cov_slib = np.outer(self.H_cor * slib_intp, self.H_cor * slib_intp)
        cov_sps = np.outer(self.H_cor * sps_intp, self.H_cor * sps_intp)
        cov_spsooo = np.outer(self.H_cor * spsooo_intp, self.H_cor * spsooo_intp)

        # suggested combination
        self.cov_mat_cor = cov_spsooo + cov_imf + cov_met + cov_stat

        self.inv_cov_cor = np.linalg.pinv(self.cov_mat_cor)

        self.data_size = self.N_unc + self.N_cor
        self.produce_residuals = True

    def get_requirements(self):
        return {
            "H_cc_unc": self.z_unc,
            "H_cc_cor": self.z_cor
        }
    
    def lnlike(self, theta, theory):
        H_unc_th = theory.get("H_cc_unc")["values"]
        H_cor_th = theory.get("H_cc_cor")["values"]

        res_unc = self.H_unc - H_unc_th
        chi2_unc = np.sum( (res_unc / self.H_err_unc)**2 )

        res_cor = self.H_cor - H_cor_th
        chi2_cor = res_cor @ self.inv_cov_cor @ res_cor 

        chi2 = chi2_unc + chi2_cor

        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2
    
    def norm_term(self):
        return 0.0
    
    def get_theory_components(self, theta, theory, z_override=None):
        x = np.concatenate([self.z_unc, self.z_cor])
        d_vec = np.concatenate([self.H_unc, self.H_cor])

        H_unc_th = theory.get("H_cc_unc")["values"]
        H_cor_th = theory.get("H_cc_cor")["values"]
        th_vec = np.concatenate([H_unc_th, H_cor_th])

        sigma = np.concatenate([self.H_err_unc, self.H_err_cor])

        return {"H": (x, d_vec, th_vec, sigma)}
    
    def get_plots(self):
        return {
            "H": True
        }
    
    def plot_constituents(self):
        z = np.concatenate([self.z_unc, self.z_cor])
        H = np.concatenate([self.H_unc, self.H_cor])
        H_err = np.concatenate([self.H_err_unc, self.H_err_cor])
        
        return z, H, H_err