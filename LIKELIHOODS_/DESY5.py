# DESY5
"""
DES-SN5YR likelihood  (Vincenzi et al. 2024 / Abbott et al. 2024)
— LEGACY dataset: 1829 SNe from the original DES-SN5YR paper release —

Data files (DATA_/DES-SN5YR/4_DISTANCES_COVMAT/)
    DES-SN5YR_HD+MetaData.csv  — 1829 distance moduli, redshifts, metadata
    STAT+SYS.txt.gz             — systematic-only covariance in SNANA flat-text
                                  format: first value is N, then N² matrix entries.
                                  Statistical uncertainty lives in MUERR_FINAL and
                                  is added to the diagonal:  C = C_sys + diag(σ²).

Note: the DES collaboration subsequently replaced these files in their GitHub repo
with the DES-Dovekie reanalysis (1820 SNe, npz inverse covariance).  The legacy
files are preserved locally for reproducibility.  For the current DES-Dovekie
dataset use the DESDovekie likelihood instead.

Slow-startup fix: STAT+SYS.txt.gz contains ~3.3 M whitespace-separated numbers.
np.genfromtxt is very slow on this; we decompress to a raw string and call
str.split() instead (~10× faster).  A binary .npy cache is also written next to
the txt.gz on first load so that subsequent instantiations are near-instant.

The nuisance parameter M (absolute magnitude / H0 offset) is analytically
marginalized over a flat prior using the Conley et al. (2011) formula.
M and H0 are fully degenerate; do not use this sample alone to measure H0.
"""

#------------------------------
# Preamble
#------------------------------
import gzip
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, UniformPrior
import numpy as np
import pandas as pd

_DATA_DES = Path(__file__).resolve().parent.parent / "DATA_" / "DES-SN5YR" / "4_DISTANCES_COVMAT"
Default_data_file = _DATA_DES / "DES-SN5YR_HD+MetaData.csv"
Default_cov_file  = _DATA_DES / "STAT+SYS.txt.gz"


def _load_snana_cov(txt_gz_path: Path) -> np.ndarray:
    """Load a SNANA flat-text covariance matrix from a .txt.gz file.

    Format: first whitespace-separated token is N (integer), followed by N²
    float values forming the full N×N covariance matrix.

    A binary .npy cache is written alongside the .gz on first load.  Subsequent
    calls load the cache directly (~100× faster than re-parsing the text).
    """
    cache_path = txt_gz_path.with_suffix("").with_suffix(".npy")   # strip .gz then .txt → .npy

    if cache_path.exists():
        cov_flat = np.load(cache_path)
    else:
        print(f"[DESY5] Parsing {txt_gz_path.name} (first-time load, building cache) …")
        with gzip.open(txt_gz_path, "rt") as fh:
            cov_flat = np.array(fh.read().split(), dtype=np.float64)
        np.save(cache_path, cov_flat)
        print(f"[DESY5] Cache written to {cache_path.name}")

    n = int(cov_flat[0])
    return cov_flat[1:].reshape(n, n)


class DESY5(LikelihoodBase):
    name = "D5"

    def __init__(self, pm, data_file=None, cov_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file
        if cov_file is None:
            cov_file = Default_cov_file

        self.data_file = Path(data_file)
        self.cov_file  = Path(cov_file)

        # ---- data vector ----
        data = pd.read_csv(self.data_file)
        mask = data["zHD"].values > 0.0      # keep all SNe with positive redshift
        self.mask = mask

        self.zCMB   = data["zHD"].values[mask]          # CMB-frame redshift (SNANA: zHD)
        self.zHEL   = data["zHEL"].values[mask]         # heliocentric redshift
        self.mu_obs = data["MU"].values[mask]            # observed distance modulus
        self.mu_err = data["MUERR_FINAL"].values[mask]  # per-SN statistical uncertainty

        N = len(self.mu_obs)
        self.N = N

        # ---- covariance ----
        # STAT+SYS.txt.gz contains the SYSTEMATIC-only covariance (C_sys).
        # The total covariance is C = C_sys + diag(MUERR_FINAL²).
        # See DATA_/DES-SN5YR/4_DISTANCES_COVMAT/README.md:
        #   "STATONLY.txt.gz is filled with zeros because the statistical
        #    uncertainties are included in MUERR_FINAL."
        cov      = _load_snana_cov(self.cov_file)
        mask_idx = np.where(mask)[0]
        cov      = cov[np.ix_(mask_idx, mask_idx)]
        np.fill_diagonal(cov, cov.diagonal() + self.mu_err ** 2)

        self.cov     = cov
        self.inv_cov = np.linalg.inv(cov)

        # Normalization
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise RuntimeError("DES-SN5YR (legacy): covariance matrix not positive definite.")
        self.ln_norm = -0.5 * (logdet + N * np.log(2.0 * np.pi))

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

