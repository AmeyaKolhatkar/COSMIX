"""SDSS DR16 combined BAO / BAO+ likelihood.

Two modes (selected via the ``mode`` kwarg, e.g. in input.yaml ``options``):

  ``"BAO"``   — Geometry only.  Gaussian block: BOSS DR12 LRG (z=0.38, 0.51),
                eBOSS LRG (z=0.698), eBOSS QSO (z=1.48), and MGS (z=0.15,
                hardcoded from Ross et al. 2015).  Non-Gaussian tabulated
                components: eBOSS ELG 1-D DV/rd table (z=0.845) and Lya
                auto + cross 2-D DM/rd–DH/rd grids (z=2.334).
                Analytic r_d marginalization is supported for the Gaussian
                block; the ELG and Lya components use the MAP r_d as an
                accurate saddle-point approximation.

  ``"BAO+"``  — Geometry + growth.  The Gaussian block gains fσ8 measurements
                for all five Gaussian tracers; the ELG 1-D table is replaced
                by a 3-D (DM/rd, DH/rd, fσ8) grid; Lya grids are unchanged.
                The nuisance parameter ``sigma80`` (σ₈ today) is always
                declared free—it is unconstrained in ``"BAO"`` mode and
                naturally constrained in ``"BAO+"`` mode by the fσ8 data.

Galaxy tracers (Gaussian block)
--------------------------------
  BOSS DR12 LRG  z = 0.38, 0.51   Alam et al. (2017)
  eBOSS LRG      z = 0.698        Bautista et al. (2021)
  eBOSS QSO      z = 1.48         Hou et al. (2021); Neveux et al. (2021)
  MGS            z = 0.15         BAO-only: Ross et al. (2015) hardcoded
                                   BAO+   : Howlett et al. (2015)

Tabulated tracers
-----------------
  eBOSS ELG  z = 0.845   Tamone et al. (2020); de Mattia et al. (2021)
  Lya auto   z = 2.334   du Mas des Bourboux et al. (2020)
  Lya cross  z = 2.334   du Mas des Bourboux et al. (2020)

Data files (DATA_/SDSSDR16/, eBOSS DR16 SVN v1.0.1)
------------------------------------------------------
BAO mode (geometry only)
  sdss_DR12_LRG_BAO_DMDH.txt / _covtot.txt
  sdss_DR16_LRG_BAO_DMDH.txt / _covtot.txt
  sdss_DR16_QSO_BAO_DMDH.txt / _covtot.txt
  sdss_DR16_ELG_BAO_DVtable.txt
  sdss_DR16_LYAUTO_BAO_DMDHgrid.txt
  sdss_DR16_LYxQSO_BAO_DMDHgrid.txt

BAO+ mode (geometry + growth)
  sdss_DR12_LRG_FSBAO_DMDHfs8.txt / _covtot.txt
  sdss_DR16_LRG_FSBAO_DMDHfs8.txt / _covtot.txt
  sdss_DR16_QSO_FSBAO_DMDHfs8.txt / _covtot.txt
  sdss_MGS_FSBAO_DVfs8.txt        / _covtot.txt
  sdss_DR16_ELG_FSBAO_DMDHfs8gridlikelihood.txt
  sdss_DR16_LYAUTO_BAO_DMDHgrid.txt   (same as BAO mode)
  sdss_DR16_LYxQSO_BAO_DMDHgrid.txt  (same as BAO mode)

Definitions
-----------
  D_M(z) : comoving angular diameter distance  [Mpc]
  D_H(z) : c / H(z)                            [Mpc]
  D_V(z) : [z · D_H · D_M²]^{1/3}             [Mpc]
  r_d    : sound horizon at the baryon-drag epoch (≈ 147.8 Mpc, eBOSS fid.)
  fσ₈(z) : f(z) · σ₈₀ · D(z)/D(0)

r_d marginalization
-------------------
  In ``"BAO"`` mode every element of the Gaussian theory vector is
  t_i = D_X(zᵢ)/r_d, i.e. linear in s ≡ 1/r_d.  The GaussMarg
  formalism therefore applies exactly:
      A = T^T C⁻¹ T,   B = T^T C⁻¹ d
      MAP r_d = A/B
  The ELG and Lya components are evaluated at this MAP r_d value
  (saddle-point approximation; excellent given the well-constrained
  Gaussian block).
  In ``"BAO+"`` mode the fσ₈ components break the linear-in-s structure,
  so analytic marginalization is not supported.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator

from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
from CORE_.ParameterManager_ import Parameter, UniformPrior

# ── Default data directory ─────────────────────────────────────────────────────
_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "DATA_" / "SDSSDR16"
)

# ── Helper: map data-file label string → theory quantity key ──────────────────
def _key_from_label(lbl: str) -> str:
    """Return the COSMIX theory key for a data-file label.

    ``DM_over_rs`` → ``"DM"``;  ``f_sigma8`` → ``"fsigma8"``, etc.
    """
    if lbl == "f_sigma8":
        return "fsigma8"
    # DM_over_rs, DH_over_rs, DV_over_rs, DM_over_rd, DH_over_rd, DV_over_rd
    return lbl.split("_")[0]       # "DM", "DH", or "DV"


# ══════════════════════════════════════════════════════════════════════════════
class SDSSDR16BAO(LikelihoodBase):
    """SDSS DR16 combined BAO / BAO+ likelihood.

    Parameters
    ----------
    pm : ParameterManager
    mode : {"BAO", "BAO+"}, default "BAO"
        Selects geometry-only (BAO) or geometry + growth (BAO+).
    data_dir : path-like, optional
        Path to the directory containing all SDSS DR16 data files.
        Defaults to ``DATA_/SDSSDR16/`` relative to the package root.
    """

    name = "SDSSDR16"

    # ── z_eff for tabulated tracers ───────────────────────────────────────────
    _ELG_Z_EFF = 0.845     # eBOSS ELG effective redshift
    _LYA_Z_EFF = 2.334     # Lya auto- and cross-correlation

    # ── MGS BAO-only hardcoded constraint (Ross et al. 2015) ─────────────────
    # DV(0.15)/r_d = 4.47 ± 0.168   (from SDSS DR7 MGS, z_eff = 0.15)
    _MGS_Z_EFF      = 0.15
    _MGS_DV_RD_MEAN = 4.47
    _MGS_DV_RD_VAR  = 0.168 ** 2    # σ² = 0.028224

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, pm, mode: str = "BAO", data_dir=None):
        super().__init__(pm)

        if mode not in ("BAO", "BAO+"):
            raise ValueError(
                f"[SDSSDR16BAO] 'mode' must be 'BAO' or 'BAO+', got {mode!r}."
            )
        self.mode = mode
        self.data_dir = (
            Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        )

        # Build data structures in three stages:
        self._load_gaussian_block()   # 1. Gaussian block (multi-tracer)
        self._load_elg()              # 2. ELG tabulated likelihood
        self._load_lya()              # 3. Lya auto + cross 2-D grids

        # Gaussian block + 1 ELG effective measurement + 2 Lya (auto + cross)
        self.data_size = len(self._gauss_val) + 1 + 2
        self.produce_residuals = True

    # ══════════════════════════════════════════════════════════════════════════
    # Parameter declarations
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def declare_parameters(cls):
        """Declare ``rd`` and ``sigma80`` as nuisance parameters.

        ``rd`` (sound horizon at baryon drag) is used by both modes.
        ``sigma80`` (σ₈ today) is needed only for the BAO+ fσ₈ predictions but
        is declared unconditionally so the class is self-contained.  In pure
        BAO mode it is unconstrained; users may fix it in their run config.
        """
        return [
            Parameter(
                name="rd",
                latex=r"r_d",
                prior=UniformPrior(100.0, 200.0),
                role="nuisance",
                status="fixed",
                value=147.8,        # eBOSS DR16 fiducial [Mpc]
                proposed_scale=0.5
            ),
            Parameter(
                name="sigma80",
                latex=r"\sigma_{80}",
                prior=UniformPrior(0.6, 1.2),
                role="nuisance",
                status="free",
                value=0.8,
                proposed_scale=0.01,
            ),
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Gaussian block construction
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _load_cov(path: Path, n: int) -> np.ndarray:
        """Load an n×n covariance matrix from a whitespace-delimited file.

        Handles both the *matrix* format (n rows × n columns) and the rare
        *flat* format (n² values on a single row / single column).
        """
        raw = np.loadtxt(path)
        if raw.ndim == 1:
            raw = raw.reshape(n, n)
        if raw.shape != (n, n):
            raise RuntimeError(
                f"[SDSSDR16BAO] Expected {n}×{n} covariance from {path.name}; "
                f"got shape {raw.shape}."
            )
        return raw

    def _load_tracer_data(self, dat_file: Path, cov_file: Path):
        """Load one Gaussian tracer.

        Returns
        -------
        z_arr   : (N,) float array of effective redshifts
        val_arr : (N,) float data vector (DX/rd or fσ₈)
        lbl_arr : (N,) string array of column labels from the file
        cov     : (N,N) covariance matrix
        """
        df = pd.read_csv(
            dat_file,
            sep=r"\s+",
            header=None,
            names=["z", "value", "label"],
            comment="#",
        )
        z   = df["z"].values.astype(float)
        val = df["value"].values.astype(float)
        lbl = df["label"].values          # keep as strings
        cov = self._load_cov(cov_file, len(z))
        return z, val, lbl, cov

    def _load_gaussian_block(self):
        """Build the joint Gaussian data vector and block-diagonal covariance.

        The block ordering is: BOSS DR12 | eBOSS LRG | eBOSS QSO | MGS.

        In ``"BAO"`` mode the block contains only D_X/r_d measurements;
        in ``"BAO+"`` mode it gains fσ₈ measurements interleaved with the
        geometry entries (in the order given by the official data files).

        The MGS tracer is special:
          - BAO mode  : hardcoded DV/r_d = 4.47 ± 0.168 (Ross et al. 2015)
          - BAO+ mode : loaded from ``sdss_MGS_FSBAO_DVfs8.txt`` which stores
                        entries in the order [fσ₈, DV/r_d] at z = 0.15.
        """
        d = self.data_dir
        m = self.mode

        # ── BOSS DR12 LRG ─────────────────────────────────────────────────────
        # BAO mode : 4 entries — DM/rd @ z=0.38, DH/rd @ z=0.38,
        #                        DM/rd @ z=0.51, DH/rd @ z=0.51
        # BAO+ mode: 6 entries — DM, DH, fσ₈ @ z=0.38, then DM, DH, fσ₈ @ z=0.51
        if m == "BAO":
            z1, v1, l1, c1 = self._load_tracer_data(
                d / "sdss_DR12_LRG_BAO_DMDH.txt",
                d / "sdss_DR12_LRG_BAO_DMDH_covtot.txt",
            )
        else:
            z1, v1, l1, c1 = self._load_tracer_data(
                d / "sdss_DR12_LRG_FSBAO_DMDHfs8.txt",
                d / "sdss_DR12_LRG_FSBAO_DMDHfs8_covtot.txt",
            )

        # ── eBOSS LRG ─────────────────────────────────────────────────────────
        # BAO mode : 2 entries — DM/rd, DH/rd @ z=0.698
        # BAO+ mode: 3 entries — DM/rd, DH/rd, fσ₈ @ z=0.698
        if m == "BAO":
            z2, v2, l2, c2 = self._load_tracer_data(
                d / "sdss_DR16_LRG_BAO_DMDH.txt",
                d / "sdss_DR16_LRG_BAO_DMDH_covtot.txt",
            )
        else:
            z2, v2, l2, c2 = self._load_tracer_data(
                d / "sdss_DR16_LRG_FSBAO_DMDHfs8.txt",
                d / "sdss_DR16_LRG_FSBAO_DMDHfs8_covtot.txt",
            )

        # ── eBOSS QSO ─────────────────────────────────────────────────────────
        # BAO mode : 2 entries — DM/rd, DH/rd @ z=1.48
        # BAO+ mode: 3 entries — DM/rd, DH/rd, fσ₈ @ z=1.48
        if m == "BAO":
            z3, v3, l3, c3 = self._load_tracer_data(
                d / "sdss_DR16_QSO_BAO_DMDH.txt",
                d / "sdss_DR16_QSO_BAO_DMDH_covtot.txt",
            )
        else:
            z3, v3, l3, c3 = self._load_tracer_data(
                d / "sdss_DR16_QSO_FSBAO_DMDHfs8.txt",
                d / "sdss_DR16_QSO_FSBAO_DMDHfs8_covtot.txt",
            )

        # ── MGS ───────────────────────────────────────────────────────────────
        # BAO mode : 1 entry — DV/rd @ z=0.15  (hardcoded; no official file)
        # BAO+ mode: 2 entries — [fσ₈, DV/rd] @ z=0.15  (Howlett et al. 2015)
        #            Note: fσ₈ comes FIRST in the file; covariance matches.
        if m == "BAO":
            z4 = np.array([self._MGS_Z_EFF])
            v4 = np.array([self._MGS_DV_RD_MEAN])
            l4 = np.array(["DV_over_rs"])
            c4 = np.array([[self._MGS_DV_RD_VAR]])
        else:
            z4, v4, l4, c4 = self._load_tracer_data(
                d / "sdss_MGS_FSBAO_DVfs8.txt",
                d / "sdss_MGS_FSBAO_DVfs8_covtot.txt",
            )

        # ── Concatenate all tracers ────────────────────────────────────────────
        self._gauss_z   = np.concatenate([z1, z2, z3, z4])
        self._gauss_val = np.concatenate([v1, v2, v3, v4])
        self._gauss_lbl = np.concatenate([l1, l2, l3, l4])

        # Block-diagonal covariance  (blocks are independent tracers)
        blocks = [c1, c2, c3, c4]
        N = sum(c.shape[0] for c in blocks)
        C = np.zeros((N, N))
        row = 0
        for c in blocks:
            n = c.shape[0]
            C[row : row + n, row : row + n] = c
            row += n

        self._gauss_cov     = C
        self._gauss_inv_cov = np.linalg.inv(C)

        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            raise RuntimeError(
                "[SDSSDR16BAO] Gaussian block covariance is not positive-definite."
            )
        self._gauss_ln_norm = -0.5 * (logdet + N * np.log(2.0 * np.pi))

        # Short two-line labels for residual plots: type on top, redshift below.
        _short = {
            "DM_over_rs": "DM/rd", "DM_over_rd": "DM/rd",
            "DH_over_rs": "DH/rd", "DH_over_rd": "DH/rd",
            "DV_over_rs": "DV/rd", "DV_over_rd": "DV/rd",
            "f_sigma8":   r"f$\sigma_8$",
        }
        self._param_labels = [
            f"{_short.get(l, l)}\nz={z:.2f}"
            for z, l in zip(self._gauss_z, self._gauss_lbl)
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — ELG tabulated likelihood
    # ══════════════════════════════════════════════════════════════════════════

    def _load_elg(self):
        """Load the eBOSS ELG tabulated likelihood.

        BAO mode : 2-column table ``(DV/rd, relative_likelihood)``.
                   Interpolated linearly in log-likelihood space.
                   Grid bounds return −∞ so out-of-range theory points are
                   automatically rejected.

        BAO+ mode: 4-column grid ``(DM/rd, DH/rd, fσ₈, relative_likelihood)``.
                   Built into a ``RegularGridInterpolator`` in log-likelihood
                   space after sorting and reshaping (see ``_load_elg_3d``).

        The relative likelihood is normalised so that its maximum equals 1
        before taking the logarithm, giving log-likelihood offset = 0 at peak.
        """
        if self.mode == "BAO":
            tab   = np.loadtxt(self.data_dir / "sdss_DR16_ELG_BAO_DVtable.txt")
            dv_rd = tab[:, 0]
            lik   = tab[:, 1]
            lik   = np.clip(lik / lik.max(), 1e-300, 1.0)
            self._elg_interp = interp1d(
                dv_rd, np.log(lik),
                kind="linear",
                bounds_error=False,
                fill_value=-np.inf,
            )
        else:
            self._load_elg_3d()

    def _load_elg_3d(self):
        """Build the 3-D ELG ``RegularGridInterpolator`` for BAO+ mode.

        The 4-column file ``sdss_DR16_ELG_FSBAO_DMDHfs8gridlikelihood.txt``
        stores a regular grid with DM/rd as the slowest-varying axis, DH/rd
        as the middle axis, and fσ₈ as the fastest-varying (innermost) axis.

        The data is lexicographically sorted before reshaping to guarantee
        the axes are strictly increasing, as required by
        ``RegularGridInterpolator``.
        """
        data = np.loadtxt(
            self.data_dir / "sdss_DR16_ELG_FSBAO_DMDHfs8gridlikelihood.txt"
        )

        # Sort: DM (axis 0, slowest), DH (axis 1), fσ₈ (axis 2, fastest)
        order = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
        data  = data[order]

        dm_vals  = np.unique(data[:, 0])
        dh_vals  = np.unique(data[:, 1])
        fs8_vals = np.unique(data[:, 2])
        n_dm, n_dh, n_fs8 = len(dm_vals), len(dh_vals), len(fs8_vals)

        if len(data) != n_dm * n_dh * n_fs8:
            raise RuntimeError(
                f"[SDSSDR16BAO] ELG 3-D grid: expected {n_dm}×{n_dh}×{n_fs8} "
                f"= {n_dm*n_dh*n_fs8} rows but got {len(data)}. "
                "Grid may not be regular."
            )

        lik    = np.clip(data[:, 3] / data[:, 3].max(), 1e-300, 1.0)
        ln_lik = np.log(lik).reshape(n_dm, n_dh, n_fs8)

        self._elg_3d_interp = RegularGridInterpolator(
            (dm_vals, dh_vals, fs8_vals), ln_lik,
            method="linear",
            bounds_error=False,
            fill_value=-np.inf,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3 — Lya 2-D grids
    # ══════════════════════════════════════════════════════════════════════════

    def _load_lya(self):
        """Load Lya auto-correlation and QSO cross-correlation grids.

        Both files share the format:
          header line: ``# D_M(z=2.334)/r_d   D_H(z=2.334)/r_d   likelihood ratio``
          data rows  : ``dm_val   dh_val   rel_lik``

        The 2-D regular grid has DM/rd as the outer axis (slow) and DH/rd
        as the inner axis (fast).  After lexsort + reshape the grid is
        stored as a ``RegularGridInterpolator`` in log-likelihood space.
        """
        for attr, fname in (
            ("_lya_auto_interp",  "sdss_DR16_LYAUTO_BAO_DMDHgrid.txt"),
            ("_lya_cross_interp", "sdss_DR16_LYxQSO_BAO_DMDHgrid.txt"),
        ):
            data  = np.loadtxt(self.data_dir / fname, comments="#")

            # Sort: DM outer (axis 0), DH inner (axis 1)
            order = np.lexsort((data[:, 1], data[:, 0]))
            data  = data[order]

            dm_vals = np.unique(data[:, 0])
            dh_vals = np.unique(data[:, 1])
            n_dm, n_dh = len(dm_vals), len(dh_vals)

            if len(data) != n_dm * n_dh:
                raise RuntimeError(
                    f"[SDSSDR16BAO] {fname}: irregular grid detected "
                    f"({n_dm}×{n_dh} = {n_dm*n_dh} expected, got {len(data)})."
                )

            lik    = np.clip(data[:, 2] / data[:, 2].max(), 1e-300, 1.0)
            ln_lik = np.log(lik).reshape(n_dm, n_dh)

            setattr(self, attr, RegularGridInterpolator(
                (dm_vals, dh_vals), ln_lik,
                method="linear",
                bounds_error=False,
                fill_value=-np.inf,
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # Analytic r_d marginalization  (BAO mode only)
    # ══════════════════════════════════════════════════════════════════════════

    def supports_marginalization(self, name: str) -> bool:
        """Analytic r_d marginalization is valid only in pure-geometry BAO mode.

        In BAO+ mode the fσ₈ components break the linear-in-(1/r_d) structure
        of the theory vector, so analytic marginalization does not apply.
        """
        return name == "rd" and self.mode == "BAO"

    def _raw_theory_vec(self, theory) -> np.ndarray:
        """Return D_X(z) values in Gaussian-block order **without** dividing by r_d.

        This is the vector T such that the theory prediction for each entry is
        ``t_i = T_i / r_d``, i.e. T is the geometry without the r_d scaling.
        Used only in BAO mode (no fσ₈ entries exist in that case).
        """
        T = []
        for z, lbl in zip(self._gauss_z, self._gauss_lbl):
            key = _key_from_label(lbl)   # "DM", "DH", or "DV"
            T.append(theory.eval(key, np.array([z]))[0])
        return np.array(T)

    def _map_rd(self, theory) -> float:
        """MAP estimate of r_d from the Gaussian block.

        Derived by completing the square in the quadratic chi² for a flat
        prior over s = 1/r_d:
            A = T^T C⁻¹ T,   B = T^T C⁻¹ d
            s_MAP = B/A  →  r_d_MAP = A/B
        """
        T = self._raw_theory_vec(theory)
        A = float(T @ self._gauss_inv_cov @ T)
        B = float(T @ self._gauss_inv_cov @ self._gauss_val)
        return A / B

    def marginalization_terms(self, theory):
        """Return the GaussMargTerm for r_d when marginalization is active.

        The framework accumulates these terms to compute the log-likelihood
        contribution from the marginalized parameter:
            ln L_marg ∝ B²/(2A)  with  A = T^T C⁻¹ T,  B = T^T C⁻¹ d.
        """
        if not self.pm.is_marginalized("rd"):
            return None
        T = self._raw_theory_vec(theory)
        A = float(T @ self._gauss_inv_cov @ T)
        B = float(T @ self._gauss_inv_cov @ self._gauss_val)
        return GaussMargTerm("rd", A, B, ln_norm=0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # Theory requirements
    # ══════════════════════════════════════════════════════════════════════════

    def get_requirements(self) -> dict:
        """Collect all theory quantities and their required redshifts.

        BAO mode requests: DM, DH at all Gaussian and Lya redshifts;
                           DV at the MGS and ELG redshifts.
        BAO+ mode adds:    fsigma8 at all Gaussian and ELG redshifts.

        Returns
        -------
        dict mapping quantity name → sorted numpy array of redshifts.
        """
        req = {}

        # ── Gaussian block ────────────────────────────────────────────────────
        for z, lbl in zip(self._gauss_z, self._gauss_lbl):
            key = _key_from_label(lbl)     # "DM", "DH", "DV", or "fsigma8"
            req.setdefault(key, set()).add(float(z))

        # ── ELG ──────────────────────────────────────────────────────────────
        z_elg = self._ELG_Z_EFF
        if self.mode == "BAO":
            req.setdefault("DV", set()).add(z_elg)
        else:
            for key in ("DM", "DH", "fsigma8"):
                req.setdefault(key, set()).add(z_elg)

        # ── Lya auto + cross (both modes use DM, DH at z=2.334) ──────────────
        z_lya = self._LYA_Z_EFF
        for key in ("DM", "DH"):
            req.setdefault(key, set()).add(z_lya)

        return {k: np.array(sorted(v)) for k, v in req.items()}

    # ══════════════════════════════════════════════════════════════════════════
    # Internal: build full Gaussian theory vector
    # ══════════════════════════════════════════════════════════════════════════

    def _gauss_theory_vec(self, theta, theory, rd: float) -> np.ndarray:
        """Build the full Gaussian-block theory vector at the given r_d.

        Geometry entries (DM/rd, DH/rd, DV/rd) are divided by ``rd``.
        Growth entries (fσ₈) are taken directly from theory.
        """
        th = []
        for z, lbl in zip(self._gauss_z, self._gauss_lbl):
            if lbl == "f_sigma8":
                th.append(theory.eval("fsigma8", np.array([z]))[0])
            else:
                key = _key_from_label(lbl)     # "DM", "DH", or "DV"
                th.append(theory.eval(key, np.array([z]))[0] / rd)
        return np.array(th)

    # ══════════════════════════════════════════════════════════════════════════
    # Log-likelihood
    # ══════════════════════════════════════════════════════════════════════════

    def lnlike(self, theta, theory) -> float:
        """Evaluate the total log-likelihood.

        The total is the sum of four independent contributions:

        1. **Gaussian block** (BOSS DR12 + eBOSS LRG + QSO + MGS):
           Standard Gaussian chi² or, when ``rd`` is analytically marginalised,
           the cosmology-independent data term −½ d^T C⁻¹ d (the GaussMargTerm
           carries the cosmological information A and B separately).

        2. **ELG** (z=0.845):
           BAO mode  — log of 1-D DV/rd interpolation.
           BAO+ mode — log of 3-D (DM/rd, DH/rd, fσ₈) interpolation.

        3. **Lya auto** (z=2.334): log of 2-D (DM/rd, DH/rd) interpolation.

        4. **Lya cross** (z=2.334): same 2-D grid, independent measurement.

        The ELG and Lya components use ``rd_eff`` — the sampled value when
        ``rd`` is a free parameter, or the MAP estimate from the Gaussian
        block when ``rd`` is analytically marginalised.
        """
        # ── 1. Gaussian block ─────────────────────────────────────────────────
        if self.pm.is_marginalized("rd"):
            # Only the data term; the GaussMargTerm carries A and B.
            chi2_d = float(
                self._gauss_val @ self._gauss_inv_cov @ self._gauss_val
            )
            ln    = -0.5 * chi2_d + self._gauss_ln_norm
            # MAP r_d used as a saddle-point approximation for ELG and Lya.
            rd_eff = self._map_rd(theory)
        else:
            rd_eff = self.pm.get_value(theta, "rd")
            th_vec = self._gauss_theory_vec(theta, theory, rd_eff)
            res    = self._gauss_val - th_vec
            chi2   = float(res @ self._gauss_inv_cov @ res)
            ln     = -0.5 * chi2 + self._gauss_ln_norm

        # ── 2. ELG ───────────────────────────────────────────────────────────
        if self.mode == "BAO":
            dv_th = theory.eval("DV", np.array([self._ELG_Z_EFF]))[0]
            ln   += float(self._elg_interp(dv_th / rd_eff))
        else:
            dm_e   = theory.eval("DM",      np.array([self._ELG_Z_EFF]))[0]
            dh_e   = theory.eval("DH",      np.array([self._ELG_Z_EFF]))[0]
            fs8_e  = theory.eval("fsigma8", np.array([self._ELG_Z_EFF]))[0]
            pt_3d  = np.array([[dm_e / rd_eff, dh_e / rd_eff, fs8_e]])
            ln    += float(self._elg_3d_interp(pt_3d)[0])

        # ── 3 & 4. Lya auto + cross ───────────────────────────────────────────
        dm_l  = theory.eval("DM", np.array([self._LYA_Z_EFF]))[0]
        dh_l  = theory.eval("DH", np.array([self._LYA_Z_EFF]))[0]
        pt_2d = np.array([[dm_l / rd_eff, dh_l / rd_eff]])
        ln   += float(self._lya_auto_interp(pt_2d)[0])
        ln   += float(self._lya_cross_interp(pt_2d)[0])

        return ln

    # ══════════════════════════════════════════════════════════════════════════
    # Residual visualization
    # ══════════════════════════════════════════════════════════════════════════

    def get_theory_components(self, theta, theory) -> dict:
        """Return the Gaussian-block data and theory for residual plots.

        The returned dict has key ``"BAO"`` with a 4-tuple
        ``(x, data_vec, theory_vec, sigma)`` where ``x`` is an integer index
        array (0, 1, …, N−1) and ``self._param_labels`` provides the tick labels
        for the Visualization layer (``z=0.380 DM/rd``, etc.).

        Returns an empty dict if the theory evaluation fails.
        """
        try:
            if self.pm.is_marginalized("rd"):
                rd = self._map_rd(theory)
            else:
                rd = self.pm.get_value(theta, "rd")

            th_vec = self._gauss_theory_vec(theta, theory, rd)
            sigma  = np.sqrt(np.diag(self._gauss_cov))
            x      = np.arange(len(self._gauss_val), dtype=float)

            return {"BAO": (x, self._gauss_val, th_vec, sigma)}
        except Exception:
            return {}
