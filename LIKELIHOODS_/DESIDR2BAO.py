"""DESIDRIIBAO — DESI DR2 BAO likelihood (full GC combination).

Implements a Gaussian likelihood for the DESI DR2 Baryon Acoustic
Oscillation measurements from the combined Galaxy Clustering sample.

The data vector contains D_M/r_d, D_H/r_d, and D_V/r_d at multiple
effective redshifts, with a full covariance matrix.

Data files (DATA_/DDTB/)
    desi_gaussian_bao_ALL_GCcomb_mean.txt — measurements
    desi_gaussian_bao_ALL_GCcomb_cov.txt  — covariance matrix

>>>The sound horizon r_d is computed analytically from the background
model using the Eisenstein-Hu approximation.<<<
"""
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, GaussianPrior, UniformPrior
import numpy as np
import pandas as pd

_DATA_DDTB = Path(__file__).resolve().parent.parent / "DATA_" / "DDTB"
Default_data_file = _DATA_DDTB / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
Default_cov_file  = _DATA_DDTB / "desi_gaussian_bao_ALL_GCcomb_cov.txt"


#------------------------------
# DDTB Class Skeleton 
#------------------------------
class DESIDRIIBAO(LikelihoodBase):
    name="DDTB"

    def __init__(self, pm, data_file=None, cov_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file
        if cov_file is None:
            cov_file = Default_cov_file

        data = pd.read_csv(data_file, sep=r"\s+", header=None, names=["z", "value", "label"])
        self.z = data["z"].values
        self.qty = data["value"].values
        self.qty_name = data["label"].values
        self.data_size = len(self.z)
        cov = np.loadtxt(cov_file).reshape(self.data_size, self.data_size)
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)

        # precompute normalization
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise RuntimeError("[DESIDRIIBAO] Covariance matrix not positive definite.")
        
        self.data_size = len(self.z)
        self.ln_norm = -0.5 * (logdet + len(self.z) * np.log(2.0 * np.pi))
        self.produce_residuals = True

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="rd",
                latex=r'r_d',
                prior=UniformPrior(100.0, 200.0),
                role="nuisance",
                status="fixed",
                value=147.05,
                proposed_scale=0.5                     # DESI DR2 BAO Paper : r_d = 147.05 Mpc
            )
        ]
        
    def get_requirements(self):
        req = {}

        for label, z in zip(self.qty_name, self.z):
            key = label.split("_")[0]
            req.setdefault(key, []).append(z)

        return {
            k: np.unique(v) for k, v in req.items()
        }
    
    def norm_term(self):
        return float(self.ln_norm)

    # ------------------------------------------------------------------ #
    # Analytic marginalisation over r_d                                   #
    # ------------------------------------------------------------------ #
    # The theory vector is linear in s = 1/r_d:                          #
    #   t_i(s) = D_X(z_i) * s                                            #
    # so the GaussMarg formalism applies exactly:                         #
    #   A = T^T C^{-1} T,  B = T^T C^{-1} d                             #
    #   MAP s = B/A  =>  MAP r_d = A/B                                   #
    # ------------------------------------------------------------------ #

    def supports_marginalization(self, name):
        return name == "rd"

    def _raw_theory_vec(self, theory):
        """D_X(z) values in data-vector order — no division by r_d."""
        T = []
        for z, label in zip(self.z, self.qty_name):
            key = label.split("_")[0]          # DV, DM, or DH
            T.append(theory.eval(key, np.array([z]))[0])
        return np.array(T)

    def _map_rd(self, theory):
        """MAP estimate of r_d when it is analytically marginalised."""
        T = self._raw_theory_vec(theory)
        A = float(T @ self.inv_cov @ T)
        B = float(T @ self.inv_cov @ self.qty)
        return A / B                            # r_d = 1 / s_MAP = A / B

    def marginalization_terms(self, theory):
        T = self._raw_theory_vec(theory)
        A = float(T @ self.inv_cov @ T)
        B = float(T @ self.inv_cov @ self.qty)
        return GaussMargTerm("rd", A, B, ln_norm=0.0)

    def lnlike(self, theta, theory):
        if self.pm.is_marginalized("rd"):
            # Cosmology-independent data term only; GaussMargTerm carries
            # all cosmological information through A and B.
            chi2_data = float(self.qty @ self.inv_cov @ self.qty)
            return -0.5 * chi2_data + self.ln_norm

        rd = self.pm.get_value(theta, "rd")

        th_list = []
        for z, label in zip(self.z, self.qty_name):
            if label == "DV_over_rs":
                val = theory.eval("DV", np.array([z]))[0] / rd
            elif label == "DM_over_rs":
                val = theory.eval("DM", np.array([z]))[0] / rd
            elif label == "DH_over_rs":
                val = theory.eval("DH", np.array([z]))[0] / rd
            else:
                raise ValueError(f"Unknown BAO observable label: {label}")
            th_list.append(val)

        th_vec = np.array(th_list)
        res = self.qty - th_vec
        chi2 = res @ self.inv_cov @ res
        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2 + self.ln_norm

    def get_theory_components(self, theta, theory, z_override=None):
        rd = self._map_rd(theory) if self.pm.is_marginalized("rd") else self.pm.get_value(theta, "rd")

        if z_override is None:
            x = self.z
            th_list = []

            for z, label in zip(self.z, self.qty_name):
                key = label.split("_")[0]
                val = theory.eval(key, np.array([z]))
                if val is None:
                    return {}
                th_list.append(val[0] / rd)

            th_vec = np.array(th_list)
            d_vec = self.qty
            sigma = np.sqrt(np.diag(self.cov))

            return {"BAO": (x, d_vec, th_vec, sigma)}

        else:
            x = z_override
            out = {}

            DV = theory.eval("DV", x)
            if DV is not None:
                out["DV"] = (x, None, DV / rd, None)

            DM = theory.eval("DM", x)
            if DM is not None:
                out["DM"] = (x, None, DM / rd, None)

            DH = theory.eval("DH", x)
            if DH is not None:
                out["DH"] = (x, None, DH / rd, None)

            return out