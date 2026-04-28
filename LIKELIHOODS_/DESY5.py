# DESY5
"""
DES-SN5YR / DES-Dovekie likelihood  (Vincenzi et al. 2024 / Abbott et al. 2024)

Data files (DATA_/DES-SN5YR/4_DISTANCES_COVMAT/)
    DES-Dovekie_HD.csv  — 1820 distance moduli, redshifts (updated DES-Dovekie sample)
    STAT+SYS.npz         — TOTAL (stat+sys) INVERSE covariance, stored as the upper
                           triangle of a 1820×1820 matrix (keys at index 0: n_sn,
                           index 1: upper-triangular values).  This is the format
                           used by the DES-SN5YR repo after the DES-Dovekie update.

Note: the DES collaboration updated the DES-SN5YR data release in early 2025 to
reflect the DES-Dovekie reanalysis (corrected distances, 1820 SNe vs. original 1829).
The old files (DES-SN5YR_HD+MetaData.csv, STAT+SYS.txt.gz) no longer exist in the
upstream repository.  DESY5 now tracks this updated dataset.

The nuisance parameter M (absolute magnitude / H0 offset) is analytically
marginalized over a flat prior using the Conley et al. (2011) formula.
M and H0 are fully degenerate; do not use this sample alone to measure H0.
"""

#------------------------------
# Preamble
#------------------------------
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, UniformPrior
import numpy as np
import pandas as pd

_DATA_DES = Path(__file__).resolve().parent.parent / "DATA_" / "DES-SN5YR" / "4_DISTANCES_COVMAT"
Default_data_file = _DATA_DES / "DES-Dovekie_HD.csv"
Default_cov_file  = _DATA_DES / "STAT+SYS.npz"

def _read_snana_hd(filepath):
    """Parse a SNANA-format Hubble diagram file into a pandas DataFrame.

    The DES-Dovekie data release uses SNANA format:
    - Lines starting with '#' are comments
    - 'VARNAMES: col1 col2 ...' defines the column names
    - 'SN: val1 val2 ...' lines are data rows
    """
    col_names = None
    rows = []
    with open(filepath, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('VARNAMES:'):
                col_names = line.split()[1:]
            elif line.startswith('SN:'):
                rows.append(line.split()[1:])
    if col_names is None:
        raise RuntimeError(f"No VARNAMES line found in {filepath}")
    df = pd.DataFrame(rows, columns=col_names)
    for col in ['zHD', 'zHEL', 'MU', 'MUERR']:
        df[col] = pd.to_numeric(df[col])
    return df


class DESY5(LikelihoodBase):
    name = "D5"

    def __init__(self, pm, data_file=None, cov_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file
        if cov_file is None:
            cov_file = Default_cov_file

        self.data_file = data_file
        self.cov_file  = cov_file

        # ---- data vector ----
        data = _read_snana_hd(self.data_file)
        mask = data["zHD"].values > 0.0      # keep all SNe with positive redshift
        self.mask = mask

        self.zCMB   = data["zHD"].values[mask]   # CMB-frame redshift (SNANA: zHD)
        self.zHEL   = data["zHEL"].values[mask]  # heliocentric redshift
        self.mu_obs = data["MU"].values[mask]     # observed distance modulus
        # Note: column is now MUERR (not MUERR_FINAL as in the pre-Dovekie release)
        self.mu_err = data["MUERR"].values[mask]  # per-SN distance uncertainty

        N = len(self.mu_obs)
        self.N = N

        # ---- covariance ----
        # STAT+SYS.npz stores the TOTAL (stat+sys) INVERSE covariance as the upper
        # triangle of the full N×N matrix.  The DES-Dovekie update changed both the
        # file format (txt.gz → npz) and the encoding (covariance → inverse covariance).
        # Positional key access matches the official DES likelihood code exactly.
        d        = np.load(self.cov_file)
        n_cov    = int(d[d.files[0]][0])
        inv_cov_full = np.zeros((n_cov, n_cov), dtype=float)
        inv_cov_full[np.triu_indices(n_cov)] = d[d.files[1]].astype(float)
        i_lower = np.tril_indices(n_cov, -1)
        inv_cov_full[i_lower] = inv_cov_full.T[i_lower]   # symmetrize

        # Apply boolean mask
        mask_idx     = np.where(mask)[0]
        self.inv_cov = inv_cov_full[np.ix_(mask_idx, mask_idx)]

        # Full covariance (needed for per-point error bars in residual plots)
        self.cov = np.linalg.inv(self.inv_cov)

        # Normalization: logdet(C) = −logdet(C⁻¹), avoids redundant inversion.
        sign_inv, logdet_inv = np.linalg.slogdet(self.inv_cov)
        if sign_inv <= 0:
            raise RuntimeError("DES-SN5YR: inverse covariance matrix not positive definite.")
        self.ln_norm = 0.5 * logdet_inv - 0.5 * N * np.log(2.0 * np.pi)

        self.ones = np.ones(N)
        self.data_size = N
        self.produce_residuals = True

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="M",
                latex=r'\mathcal{M}',
                prior=UniformPrior(-19.6, -18.8),
                role="nuisance",
                status="marginalized",
                proposed_scale=0.001
            )
        ]

    def get_requirements(self):
        return {"dL": self.zCMB}

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
        factor = (1.0 + self.zHEL) / (1.0 + self.zCMB)
        mu_th = 5.0 * np.log10(factor * dL) + 25.0

        if self.pm.is_marginalized("M"):
            res = self.mu_obs - mu_th
        else:
            M = self.pm.get_value(theta, "M")
            res = self.mu_obs - mu_th - M

        chi2 = float(res @ self.inv_cov @ res)
        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2 + self.ln_norm

    def norm_term(self):
        return float(self.ln_norm)

    def _map_M(self, mu_th_noM):
        """MAP value of M when analytically marginalized (B/A from Gaussian integral)."""
        r = self.mu_obs - mu_th_noM
        A = float(self.ones @ self.inv_cov @ self.ones)
        B = float(self.ones @ self.inv_cov @ r)
        return B / A

    def get_theory_components(self, theta, theory, z_override=None):
        dL_data = theory.eval("dL", self.zCMB)
        factor = (1.0 + self.zHEL) / (1.0 + self.zCMB)
        mu_th_noM = 5.0 * np.log10(factor * dL_data) + 25.0

        if self.pm.is_marginalized("M"):
            M = self._map_M(mu_th_noM)
        else:
            M = self.pm.get_value(theta, "M")

        if z_override is None:
            sigma = np.sqrt(np.diag(self.cov))
            return {"mu": (self.zCMB, self.mu_obs, mu_th_noM + M, sigma)}
        else:
            dL = theory.eval("dL", z_override)
            mu_th = 5.0 * np.log10(dL) + 25.0
            return {"mu": (z_override, None, mu_th + M, None)}

    def get_plots(self):
        return {"mu": True}

