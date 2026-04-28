"""DatasetConsistency — dataset concordance / tension estimators.

Implements two statistical tools from de Cruz Perez, Park & Ratra (2024),
arXiv:2404.19194, Section III, for quantifying whether two datasets are
mutually consistent when constraining a given cosmological model.

Both estimators require *three* completed runs on the same model:
    run_1  — sampler run using dataset D_1 only
    run_2  — sampler run using dataset D_2 only
    run_12 — sampler run using the combined dataset D_1 ∪ D_2

──────────────────────────────────────────────────────────────────────────
ESTIMATOR 1 — DIC-based consistency  log10(I)   [Eq. 12, arXiv:2404.19194]
──────────────────────────────────────────────────────────────────────────
    G   = DIC(D_12) - DIC(D_1) - DIC(D_2)
    I   = exp(-G / 2)
    log10(I) > 0  →  datasets consistent
    log10(I) < 0  →  datasets inconsistent

Jeffreys' scale:
    |log10(I)| > 0.5  substantial
    |log10(I)| > 1.0  strong
    |log10(I)| > 2.0  decisive

Works for ANY sampler (emcee or Dynesty) since it only needs DIC values
from diagnostics.json.

──────────────────────────────────────────────────────────────────────────
ESTIMATOR 2 — Suspiciousness / tension probability  [Eqs. 15-16]
──────────────────────────────────────────────────────────────────────────
Based on Handley & Lemos (2019b), arXiv:1902.04029.

    Bayes ratio       ln R  = ln Z_12 - ln Z_1 - ln Z_2
    KL divergence     D_D   = <ln L>_posterior - ln Z
    Information ratio ln I  = D_1 + D_2 - D_12
    Suspiciousness    ln S  = ln R - ln I
    Bayesian dim      d     = d̃_1 + d̃_2 - d̃_12  (≈ n_free_params)
                      d̃/2  = Var_posterior[ln L]

Under concordance: -2 ln S ~ chi2(d), so
    p     = P(chi2_d > -2 ln S)            (tension probability)
    sigma = sqrt(2) * erfcinv(1 - p)       (Gaussian sigma equivalent)

Thresholds:
    p < 0.05  (sigma ≈ 2)  moderate tension
    p < 0.003 (sigma ≈ 3)  strong tension

REQUIRES Dynesty nested-sampling runs (needs logZ and weights.npy).
For emcee runs, use the DIC-based estimator only.

──────────────────────────────────────────────────────────────────────────
Usage (from DatasetTension.ipynb)
──────────────────────────────────────────────────────────────────────────
    from POST_PROCESSING_.DatasetConsistency import RunLoader, DatasetConsistency

    run1  = RunLoader.load("RUNS_/run_LCDM_CC_dynesty_1")
    run2  = RunLoader.load("RUNS_/run_LCDM_PP_dynesty_1")
    run12 = RunLoader.load("RUNS_/run_LCDM_CC_PP_dynesty_1")

    log10_I, G, label = DatasetConsistency.log10_I(run1, run2, run12)
    result = DatasetConsistency.suspiciousness_and_tension(run1, run2, run12)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.special import erfcinv


# ──────────────────────────────────────────────────────────────────────────────
# RunLoader
# ──────────────────────────────────────────────────────────────────────────────

class RunLoader:
    """Load a completed COSMIX run from its archive directory."""

    @staticmethod
    def load(run_dir: str | Path) -> dict:
        """
        Load chain, log_prob, weights, and diagnostics from a run directory.

        Returns a dict with keys:
            run_dir       (Path)
            chain         (ndarray, shape [N, n_params])
            log_prob      (ndarray, shape [N]) — raw log likelihood
            weights       (ndarray or None)    — posterior weights (nested only)
            DIC           (float)              — from diagnostics.json
            AIC           (float)
            chi2_min      (float or None)
            logZ_raw      (float or None)      — raw Dynesty logZ
            logZ_physical (float or None)      — logZ with norm subtracted
            logZ_err      (float or None)
            norm_terms    (float or None)      — likelihood normalization sum
            param_names   (list[str])
            likelihoods   (list[str])
            sampler_mode  (str)                — "nested" or "mcmc"
        """
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # --- chains -----------------------------------------------------------
        chain    = np.load(run_dir / "chain.npy")
        log_prob = np.load(run_dir / "log_prob.npy")

        weights_path = run_dir / "weights.npy"
        if weights_path.exists():
            weights = np.load(weights_path)
            weights = weights / weights.sum()   # ensure normalized
        else:
            weights = None

        # --- diagnostics ------------------------------------------------------
        diag_path = run_dir / "diagnostics.json"
        if not diag_path.exists():
            raise FileNotFoundError(f"diagnostics.json not found in {run_dir}")
        with open(diag_path) as f:
            diag = json.load(f)

        ic = diag.get("information_criteria") or {}
        DIC      = ic.get("DIC")
        AIC      = ic.get("AIC")
        chi2_min = ic.get("chi2_min")

        # logZ: prefer explicit keys added in newer COSMIX; fall back to legacy "logZ"
        logZ_raw      = diag.get("logZ_raw")      or diag.get("logZ")
        logZ_physical = diag.get("logZ_physical") or diag.get("logZ")
        logZ_err      = diag.get("logZ_err")
        norm_terms    = diag.get("norm_terms_total")

        # --- manifest ---------------------------------------------------------
        manifest_path = run_dir / "manifest.yaml"
        param_names  = []
        likelihoods  = []
        sampler_mode = "mcmc"
        if manifest_path.exists():
            try:
                import yaml
                with open(manifest_path) as f:
                    mf = yaml.safe_load(f)
                param_names = mf.get("labels", {}).get("names", [])
                likelihoods = [
                    entry.get("name", entry.get("module", "?"))
                    for entry in (mf.get("likelihoods") or [])
                ]
                conv_mode = mf.get("convergence", {}).get("mode", "mcmc")
                if conv_mode == "nested":
                    sampler_mode = "nested"
            except Exception:
                pass

        if DIC is None:
            raise ValueError(
                f"DIC not found in diagnostics.json for {run_dir}. "
                "Re-run with a newer version of COSMIX that saves information_criteria."
            )

        return dict(
            run_dir=run_dir,
            chain=chain,
            log_prob=log_prob,
            weights=weights,
            DIC=float(DIC),
            AIC=float(AIC) if AIC is not None else None,
            chi2_min=float(chi2_min) if chi2_min is not None else None,
            logZ_raw=float(logZ_raw) if logZ_raw is not None else None,
            logZ_physical=float(logZ_physical) if logZ_physical is not None else None,
            logZ_err=float(logZ_err) if logZ_err is not None else None,
            norm_terms=float(norm_terms) if norm_terms is not None else None,
            param_names=param_names,
            likelihoods=likelihoods,
            sampler_mode=sampler_mode,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Interpretation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _interpret_log10_I(val: float) -> str:
    sign = "consistent" if val >= 0 else "inconsistent"
    a = abs(val)
    if a < 0.5:
        strength = "neither substantial consistency nor inconsistency"
    elif a < 1.0:
        strength = f"substantial {sign}"
    elif a < 2.0:
        strength = f"strong {sign}"
    else:
        strength = f"decisive {sign}"
    return strength


def _interpret_tension(sigma: float, p: float) -> str:
    if p > 0.32:
        return "no significant tension  (< 1σ)"
    if p > 0.05:
        return f"mild tension  ({sigma:.2f}σ)"
    if p > 0.003:
        return f"moderate tension  ({sigma:.2f}σ, p = {100*p:.2f}%)"
    return f"strong tension  ({sigma:.2f}σ, p = {100*p:.3f}%)"


# ──────────────────────────────────────────────────────────────────────────────
# Core estimators
# ──────────────────────────────────────────────────────────────────────────────

class DatasetConsistency:
    """
    Static methods for dataset concordance / tension analysis.

    All methods accept the dicts returned by RunLoader.load().
    """

    # ------------------------------------------------------------------
    # Estimator 1: DIC-based log10(I)
    # ------------------------------------------------------------------

    @staticmethod
    def log10_I(run1: dict, run2: dict, run12: dict) -> dict:
        """
        Compute the DIC-based consistency estimator log10(I).

        Works for any sampler (emcee or Dynesty).

        Parameters
        ----------
        run1, run2, run12 : dicts from RunLoader.load()
            run1  — D_1-only run
            run2  — D_2-only run
            run12 — combined D_1 + D_2 run

        Returns
        -------
        dict with keys:
            G          DIC(D_12) - DIC(D_1) - DIC(D_2)
            log10_I    log10(exp(-G/2))  = -G / (2 ln10)
            I          exp(-G/2)
            label      human-readable interpretation
        """
        G      = run12["DIC"] - run1["DIC"] - run2["DIC"]
        log10I = -G / (2.0 * np.log(10.0))
        I      = np.exp(-G / 2.0)
        label  = _interpret_log10_I(log10I)

        return dict(G=G, log10_I=log10I, I=I, label=label)

    # ------------------------------------------------------------------
    # Estimator 2: Suspiciousness + tension probability
    # ------------------------------------------------------------------

    @staticmethod
    def suspiciousness_and_tension(run1: dict, run2: dict, run12: dict) -> dict:
        """
        Compute the suspiciousness-based tension probability.

        Requires nested-sampling runs (logZ and weights.npy).

        Theory
        ------
        KL divergence (per run):
            D_KL  = <ln L>_posterior - ln Z   = sum_i w_i * (logl_i - logZ)
            d̃/2  = Var_posterior[ln L]        = Var_w[logl]  (≈ effective parameter count)

        Bayes ratio:    ln R = ln Z_12 - ln Z_1 - ln Z_2
        Info ratio:     ln I = D_1 + D_2 - D_12
        Suspiciousness: ln S = ln R - ln I
        Bayes dim:      d    = d̃_1 + d̃_2 - d̃_12  (should be ≈ n_free_params)

        Tension p-value: -2 ln S ~ chi2(d)  under concordance
            p = P(chi2_d > -2 ln S)
            sigma = sqrt(2) * erfcinv(1 - p)

        Parameters
        ----------
        run1, run2, run12 : dicts from RunLoader.load()

        Returns
        -------
        dict with keys:
            D_KL_1, D_KL_2, D_KL_12     KL divergences
            d_tilde_1, d_tilde_2, d_tilde_12  Bayesian model complexities
            d                            Bayesian model dimensionality
            ln_R, ln_I, ln_S             log Bayes/Info/Suspiciousness ratios
            chi2_tension                 -2 ln S
            p                            tension probability
            sigma                        Gaussian sigma equivalent
            sigma_logZ_err               1-sigma uncertainty from logZ errors
            label                        human-readable interpretation
            warnings                     list of any non-fatal issues
        """
        warnings_list = []

        # --- Validate nested sampling availability ----------------------------
        for tag, run in [("D1", run1), ("D2", run2), ("D12", run12)]:
            if run["logZ_raw"] is None:
                raise ValueError(
                    f"Run {tag} ({run['run_dir']}) has no logZ.  "
                    "Suspiciousness requires Dynesty nested-sampling runs."
                )
            if run["weights"] is None and run["sampler_mode"] != "nested":
                # MCMC run: equal-weight KL estimate is a rough approximation.
                warnings_list.append(
                    f"Run {tag} ({run['run_dir']}) is an MCMC run — "
                    "KL-divergence will be estimated from equal-weight samples "
                    "(less accurate than importance-weighted nested sampling)."
                )
            # For nested runs with weights=None: samples are bootstrap-resampled
            # posterior — equal-weight mean/variance gives an unbiased KL estimate.

        def _kl_and_dim(run: dict) -> tuple[float, float]:
            """Return (D_KL, d_tilde) for one run."""
            logl  = run["log_prob"]
            logZ  = run["logZ_raw"]
            w     = run["weights"]

            if w is None:
                # Equal-weight fallback (MCMC or old nested run without weights.npy)
                w = np.ones(len(logl), dtype=float)
                w /= w.sum()
            else:
                w = w / w.sum()

            log_IS = logl - logZ                        # Shannon information
            D_KL   = float(np.sum(w * log_IS))
            # d̃ = 2 * Var_posterior[log_IS]
            d_tilde = 2.0 * float(
                np.sum(w * log_IS**2) - D_KL**2
            )
            # Guard against tiny negative values from floating-point noise
            d_tilde = max(d_tilde, 0.0)
            return D_KL, d_tilde

        D_KL_1,   d_tilde_1   = _kl_and_dim(run1)
        D_KL_2,   d_tilde_2   = _kl_and_dim(run2)
        D_KL_12,  d_tilde_12  = _kl_and_dim(run12)

        # --- Bayes / Info / Suspiciousness ratios ----------------------------
        ln_R = run12["logZ_raw"] - run1["logZ_raw"] - run2["logZ_raw"]
        ln_I = D_KL_1 + D_KL_2 - D_KL_12
        ln_S = ln_R - ln_I

        # Bayesian model dimensionality (should ≈ n_free_params as sanity check)
        d = d_tilde_1 + d_tilde_2 - d_tilde_12

        if d <= 0:
            warnings_list.append(
                f"Bayesian model dimensionality d = {d:.3f} ≤ 0.  This can "
                "happen when the combined posterior is broader than the individual "
                "ones (unusual).  The chi2 p-value is unreliable; report ln S only."
            )
            p     = float("nan")
            sigma = float("nan")
            label = "indeterminate (d ≤ 0)"
        else:
            chi2_val = float(-2.0 * ln_S)
            p        = float(chi2_dist.sf(chi2_val, df=d))

            if p <= 0.0:
                sigma = float("inf")
            elif p >= 1.0:
                sigma = 0.0
            else:
                sigma = float(np.sqrt(2.0) * erfcinv(1.0 - p))

            label = _interpret_tension(sigma, p)

        # --- logZ error propagation ------------------------------------------
        # Uncertainty on ln R from logZ measurement errors.
        # sigma(ln S) ≈ sigma(ln R) since KL terms have much smaller uncertainty.
        logZ_errs = [
            run["logZ_err"] if run["logZ_err"] is not None else 0.0
            for run in (run1, run2, run12)
        ]
        sigma_logZ_err = float(np.sqrt(sum(e**2 for e in logZ_errs)))

        return dict(
            D_KL_1=D_KL_1,
            D_KL_2=D_KL_2,
            D_KL_12=D_KL_12,
            d_tilde_1=d_tilde_1,
            d_tilde_2=d_tilde_2,
            d_tilde_12=d_tilde_12,
            d=d,
            ln_R=float(ln_R),
            ln_I=float(ln_I),
            ln_S=float(ln_S),
            chi2_tension=float(-2.0 * ln_S),
            p=p,
            sigma=sigma,
            sigma_logZ_err=sigma_logZ_err,
            label=label,
            warnings=warnings_list,
        )

    # ------------------------------------------------------------------
    # Δ AIC / Δ DIC model selection (convenience, already in ModelComparison)
    # ------------------------------------------------------------------

    @staticmethod
    def delta_IC(runs: dict[str, dict], reference: str) -> dict:
        """
        Compute ΔAIC and ΔDIC relative to a reference model run.

        Parameters
        ----------
        runs      : {model_name: run_dict}  from RunLoader.load()
        reference : key of the reference model (e.g. "flat_LCDM")

        Returns
        -------
        dict of {model_name: {"ΔAIC": float, "ΔDIC": float}}
        """
        if reference not in runs:
            raise KeyError(f"Reference model '{reference}' not in runs dict.")
        ref = runs[reference]
        out = {}
        for name, run in runs.items():
            dAIC = (run["AIC"] - ref["AIC"]) if (run["AIC"] and ref["AIC"]) else None
            dDIC = run["DIC"] - ref["DIC"]
            out[name] = {"ΔAIC": dAIC, "ΔDIC": dDIC}
        return out
