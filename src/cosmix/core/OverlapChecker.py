"""OverlapChecker — detects shared survey measurements across COSMIX likelihoods.

When two likelihoods contain measurements from the same galaxy survey at the
same redshift, their errors are correlated through the shared underlying galaxy
catalog.  Including both without a covariance term double-counts the survey's
information and can artificially tighten constraints or bias model selection.

A concrete example: VIPERS PDR-2 contributes both fσ₈ (via the RSD likelihood)
and E_G (via the EgStatistic likelihood) from the same spectroscopic catalog.
Running both likelihoods simultaneously without correction is invalid.

Typical use (pre-run check in a run script)
-------------------------------------------
    from CORE_.OverlapChecker import OverlapChecker

    checker = OverlapChecker(z_tol=0.01)
    report  = checker.check(pipeline.likelihoods)
    report.pretty_print()

    if report.has_overlaps and abort_on_overlap:
        raise RuntimeError("Overlapping datasets detected.")

Standalone use (in a Jupyter notebook)
---------------------------------------
    from CORE_.OverlapChecker import OverlapChecker
    from LIKELIHOODS_.RSD import RedshiftSpaceDistortion
    from LIKELIHOODS_.EgStatistic import EgStatistic
    from CORE_.ParameterManager_ import ParameterManager

    pm = ParameterManager()            # minimal PM — no parameters needed for data loading
    likelihoods = [RedshiftSpaceDistortion(pm), EgStatistic(pm)]
    report = OverlapChecker().check(likelihoods)
    report.pretty_print()
    df = report.to_dataframe()         # tidy DataFrame for further inspection

Extending the survey synonym dictionary
----------------------------------------
The canonical survey name is resolved by stripping whitespace / punctuation,
lowercasing, then looking up in ``SURVEY_CANONICAL``.  To handle a new survey
name variant, either add it to ``SURVEY_CANONICAL`` directly or pass it via
``extra_synonyms`` at construction time:

    checker = OverlapChecker(
        extra_synonyms={"desifullshapedr1": "DESI DR1 FS"}
    )

Keys in ``extra_synonyms`` must be lowercase and stripped of all non-alphanumeric
characters (run ``re.sub(r'[\\s\\-_+,.]+', '', name).lower()`` to generate one).
"""

from __future__ import annotations

import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ── Canonical survey name registry ────────────────────────────────────────────
#
# Keys   : normalised variant strings (lowercase, no spaces / hyphens / dots).
# Values : canonical label used for overlap comparison.
#
# Rule for adding new entries:
#   key   = re.sub(r'[\s\-_+,.]+', '', variant_name).lower()
#   value = canonical label agreed in the paper (can be any consistent string)
#
# IMPORTANT: compound survey names that appear in Eg compilations
# (e.g. "KiDS-1000+BOSS DR12+2dFLenS") are NOT broken into their
# constituent parts here.  Doing so would cause false positives when a
# standalone BOSS DR12 likelihood appears alongside a composite Eg survey.
# Only add entries for surveys that appear as a *single standalone entry*
# in at least one data file.

SURVEY_CANONICAL: dict[str, str] = {
    # ── VIPERS (same PDR-2 spectroscopic catalog feeds both RSD and Eg) ──────
    "vipers":                "VIPERS",
    "viperspdr2":            "VIPERS",
    "viperspdr-2":           "VIPERS",
    "vimosvltdeepsurvey":    "VIPERS",

    # ── BOSS / SDSS DR12 ─────────────────────────────────────────────────────
    "bossdr12":              "BOSS DR12",
    "boss":                  "BOSS DR12",
    "sdssbossdr12":          "BOSS DR12",

    # ── eBOSS / SDSS DR16 ────────────────────────────────────────────────────
    "eboss":                 "eBOSS DR16",
    "ebossdr16":             "eBOSS DR16",
    "sdsseboss":             "eBOSS DR16",

    # ── WiggleZ ───────────────────────────────────────────────────────────────
    "wigglez":               "WiggleZ",

    # ── 6dF Galaxy Survey ─────────────────────────────────────────────────────
    "6dfgs":                 "6dFGS",
    "6dfgssnia":             "6dFGS",

    # ── GAMA ─────────────────────────────────────────────────────────────────
    "gama":                  "GAMA",

    # ── FastSound ─────────────────────────────────────────────────────────────
    "fastsound":             "FastSound",

    # ── 2dFGRS ───────────────────────────────────────────────────────────────
    "2dfgrs":                "2dFGRS",

    # ── 2MASS / 2MTF ─────────────────────────────────────────────────────────
    "2mass":                 "2MASS",
    "2mtf":                  "2MTF+6dFGSv",
    "2mtf+6dfgsv":           "2MTF+6dFGSv",
    "2mtf6dfgsv":            "2MTF+6dFGSv",

    # ── ALFALFA ───────────────────────────────────────────────────────────────
    "alfalfa":               "ALFALFA",

    # ── IRAS ─────────────────────────────────────────────────────────────────
    "iras":                  "IRAS",
    "irассnia":              "IRAS",   # just in case
    "iras+snia":             "IRAS",

    # ── DESI DR1 Full-Shape ───────────────────────────────────────────────────
    "desifullshapedr1":      "DESI DR1 FS",
    "desidr1fs":             "DESI DR1 FS",
    "desi1fs":               "DESI DR1 FS",

    # ── SDSS-IV (extended BOSS) ───────────────────────────────────────────────
    "sdssiv":                "SDSS-IV",
    "sdss-iv":               "SDSS-IV",
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _make_key(survey: str) -> str:
    """Normalise a survey string to a lookup key.

    Strips all whitespace, hyphens, underscores, plus signs, commas, and
    dots, then lowercases.  Result is used to look up ``SURVEY_CANONICAL``.
    """
    return re.sub(r"[\s\-_+,\.]+", "", survey).lower()


def _resolve_canonical(survey: str, synonyms: dict[str, str]) -> str:
    """Return the canonical name for *survey* using *synonyms*."""
    return synonyms.get(_make_key(survey), survey.strip())


# ── DataPoint ─────────────────────────────────────────────────────────────────

@dataclass
class DataPoint:
    """Metadata for a single observational data point as exposed by a likelihood.

    Attributes
    ----------
    likelihood_name : str
        The ``name`` attribute of the source likelihood (e.g. ``"RSD"``, ``"Eg"``).
    observable : str
        Physical quantity being constrained (e.g. ``"fσ₈"``, ``"E_G"``, ``"H"``).
    z : float
        Effective redshift of the measurement.
    survey : str
        Raw survey label exactly as it appears in the data file.
    value : float, optional
        Central value of the measurement (for reporting only).
    error : float, optional
        1σ uncertainty (for reporting only).
    reference : str, optional
        Bibliographic reference string (for reporting only).
    """

    likelihood_name: str
    observable: str
    z: float
    survey: str
    value: Optional[float] = None
    error: Optional[float] = None
    reference: Optional[str] = None


# ── Overlap ───────────────────────────────────────────────────────────────────

@dataclass
class Overlap:
    """A pair of data points identified as sharing an underlying galaxy survey."""

    point_a: DataPoint
    point_b: DataPoint

    @property
    def delta_z(self) -> float:
        return abs(self.point_a.z - self.point_b.z)

    @property
    def summary(self) -> str:
        pa, pb = self.point_a, self.point_b
        return (
            f"[{pa.likelihood_name}] '{pa.survey}'"
            f"  z={pa.z:.4f}  ({pa.observable})"
            f"\n            ↔  [{pb.likelihood_name}] '{pb.survey}'"
            f"  z={pb.z:.4f}  ({pb.observable})"
            f"  |Δz|={self.delta_z:.4f}"
        )


# ── OverlapReport ─────────────────────────────────────────────────────────────

class OverlapReport:
    """Container and reporter for the results of an overlap check.

    Attributes
    ----------
    overlaps : list of Overlap
        All detected pairs.
    z_tol : float
        The redshift tolerance used during the check.
    likelihoods_checked : list of str
        Names of likelihoods that contributed at least one data point.
    """

    def __init__(
        self,
        overlaps: List[Overlap],
        z_tol: float,
        likelihoods_checked: List[str],
    ):
        self.overlaps = overlaps
        self.z_tol = z_tol
        self.likelihoods_checked = likelihoods_checked

    @property
    def has_overlaps(self) -> bool:
        return len(self.overlaps) > 0

    def pretty_print(self) -> None:
        """Print a human-readable overlap report to stdout."""
        sep = "─" * 72
        print(sep)
        print("  COSMIX Dataset Overlap Report")
        print(f"  Likelihoods checked : {', '.join(self.likelihoods_checked)}")
        print(f"  Redshift tolerance  : Δz < {self.z_tol}")
        print(sep)

        if not self.has_overlaps:
            print("  ✓  No overlapping survey measurements detected.\n")
        else:
            n = len(self.overlaps)
            print(f"  ✗  {n} overlap{'s' if n > 1 else ''} detected:\n")
            for i, ov in enumerate(self.overlaps, 1):
                print(f"  [{i}]  {ov.summary}")
                print()
            print("  ACTION REQUIRED")
            print("  ────────────────")
            print("  Option A (recommended): Remove the duplicate points from one")
            print("  likelihood's data file before running chains.")
            print("  Option B: Add the cross-likelihood covariance term explicitly.")
            print("  Option C: Set abort_on_overlap: false in input.yaml to proceed")
            print("  with awareness of the issue (document the caveat in your paper).")
        print(sep)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy pandas DataFrame for notebook inspection or export.

        Each row corresponds to one overlap pair.  Columns ending in ``_A``
        and ``_B`` describe the two data points respectively.
        """
        rows = []
        for ov in self.overlaps:
            pa, pb = ov.point_a, ov.point_b
            rows.append(
                {
                    "canonical_survey": _resolve_canonical(pa.survey, SURVEY_CANONICAL),
                    "likelihood_A":     pa.likelihood_name,
                    "observable_A":     pa.observable,
                    "z_A":              pa.z,
                    "survey_A":         pa.survey,
                    "value_A":          pa.value,
                    "error_A":          pa.error,
                    "likelihood_B":     pb.likelihood_name,
                    "observable_B":     pb.observable,
                    "z_B":             pb.z,
                    "survey_B":         pb.survey,
                    "value_B":          pb.value,
                    "error_B":          pb.error,
                    "delta_z":          ov.delta_z,
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"OverlapReport(n_overlaps={len(self.overlaps)}, "
            f"likelihoods={self.likelihoods_checked})"
        )


# ── OverlapChecker ────────────────────────────────────────────────────────────

class OverlapChecker:
    """Detect shared survey measurements across a list of likelihood instances.

    Parameters
    ----------
    z_tol : float
        Maximum redshift separation Δz below which two points at the same
        survey are flagged as overlapping.  Default 0.01.
    extra_synonyms : dict[str, str], optional
        Additional survey aliases not already in ``SURVEY_CANONICAL``.
        Keys must be normalised (lowercase, no spaces / punctuation).
        These are merged with ``SURVEY_CANONICAL`` for this checker instance
        only — the module-level dict is not modified.

        Example::

            checker = OverlapChecker(
                extra_synonyms={"desifullshapedr1": "DESI DR1 FS"}
            )
    """

    def __init__(
        self,
        z_tol: float = 0.01,
        extra_synonyms: Optional[dict] = None,
    ):
        self.z_tol = z_tol
        self._synonyms = {**SURVEY_CANONICAL, **(extra_synonyms or {})}

    def _canonical(self, survey: str) -> str:
        return _resolve_canonical(survey, self._synonyms)

    def check(self, likelihoods) -> OverlapReport:
        """Run the overlap check on a list of likelihood instances.

        Likelihoods that do not implement ``data_manifest()`` (or return an
        empty list) are silently skipped — the default base-class behaviour
        is appropriate for priors, CMB-compressed likelihoods, etc.

        Parameters
        ----------
        likelihoods : list
            Instantiated likelihood objects.  Each must have a
            ``data_manifest()`` method (see
            :class:`~CORE_.LikelihoodBase_.LikelihoodBase`).

        Returns
        -------
        OverlapReport
        """
        all_points: List[DataPoint] = []
        checked_names: List[str] = []

        for L in likelihoods:
            manifest = L.data_manifest()
            if manifest:
                all_points.extend(manifest)
                if L.name not in checked_names:
                    checked_names.append(L.name)

        overlaps: List[Overlap] = []

        # O(n²) pairwise scan — datasets are small (< 100 pts each)
        for i, pa in enumerate(all_points):
            for pb in all_points[i + 1 :]:
                # Skip intra-likelihood pairs
                if pa.likelihood_name == pb.likelihood_name:
                    continue
                # Check canonical survey match
                if self._canonical(pa.survey) != self._canonical(pb.survey):
                    continue
                # Check redshift proximity
                if abs(pa.z - pb.z) < self.z_tol:
                    overlaps.append(Overlap(pa, pb))

        return OverlapReport(
            overlaps=overlaps,
            z_tol=self.z_tol,
            likelihoods_checked=checked_names,
        )

    def check_from_pipeline(self, pipeline) -> OverlapReport:
        """Convenience wrapper: pass a :class:`~CORE_.Pipeline.Pipeline` directly."""
        return self.check(pipeline.likelihoods)
