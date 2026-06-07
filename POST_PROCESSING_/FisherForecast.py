"""FisherForecast — parameter-sensitivity forecasting via the Fisher information matrix.

Computes the forecasted 1σ constraints on all free model parameters given
a set of future survey specifications.  The Fisher matrix is built from
numerical central-difference derivatives of the theory predictions with
respect to each free parameter, evaluated at the fiducial point.

Supported observable types
--------------------------
``FSigma8Bin``
    One RSD measurement of fσ₈(z) ≡ f(z)·σ₈₀·D(z)/D(0).
    Uncertainty ``sigma`` is the absolute error on fσ₈.

``BAOBin``
    One BAO measurement providing D_H(z)/r_d and D_M(z)/r_d.
    Uncertainties ``frac_DH`` and ``frac_DM`` are *fractional* errors
    (e.g. 0.008 = 0.8 %).  The module converts to absolute errors using
    the fiducial theory predictions.  Pass a ``BAOBin`` with ``frac_DM=0``
    or ``frac_DH=0`` to include only one of the two distance observables.

r_d handling
------------
r_d is needed to evaluate D_H/r_d and D_M/r_d.  It is obtained in order of
precedence:
  1. From the pipeline's ParameterManager: if ``rd`` is a free parameter its
     fiducial value is used; if it is fixed the fixed value is used.
  2. From the ``rd_fiducial`` kwarg passed to ``FisherForecast.__init__``.
  3. Default: 147.78 Mpc (eBOSS DR16 fiducial).

Usage
-----
    from POST_PROCESSING_.FisherForecast import (
        FisherForecast, FSigma8Bin, BAOBin, SurveySpec
    )

    desi_proxy = SurveySpec(
        name="DESI_DR3_proxy",
        rsd_bins=[
            FSigma8Bin(z_eff=0.51,  sigma=0.020),
            FSigma8Bin(z_eff=0.706, sigma=0.017),
            FSigma8Bin(z_eff=0.930, sigma=0.013),
            FSigma8Bin(z_eff=1.317, sigma=0.015),
        ],
        bao_bins=[
            BAOBin(z_eff=0.51,  frac_DH=0.008, frac_DM=0.008),
            BAOBin(z_eff=0.706, frac_DH=0.007, frac_DM=0.007),
            BAOBin(z_eff=0.930, frac_DH=0.006, frac_DM=0.006),
            BAOBin(z_eff=1.317, frac_DH=0.008, frac_DM=0.008),
        ],
    )

    fc = FisherForecast(pipeline, fiducial_theta=theta_fid, surveys=[desi_proxy])
    result = fc.compute()
    print(result.sigma)   # {'H0': 0.21, 'Omegam0': 0.003, 'lambda0': 0.14, ...}
    result.plot_ellipse("H0", "lambda0")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ── fiducial r_d (eBOSS DR16 value) ──────────────────────────────────────────
_RD_FIDUCIAL_DEFAULT = 147.78   # Mpc


# ══════════════════════════════════════════════════════════════════════════════
# Survey specification dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FSigma8Bin:
    """One RSD measurement bin.

    Parameters
    ----------
    z_eff : float
        Effective redshift of the measurement.
    sigma : float
        Absolute 1σ uncertainty on fσ₈(z_eff).
    """
    z_eff: float
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"[FSigma8Bin] sigma must be > 0; got {self.sigma}.")


@dataclass
class BAOBin:
    """One BAO measurement bin.

    Parameters
    ----------
    z_eff : float
        Effective redshift.
    frac_DH : float
        Fractional 1σ uncertainty on D_H(z)/r_d  (e.g. 0.008 = 0.8 %).
        Set to 0 to exclude the D_H observable from this bin.
    frac_DM : float
        Fractional 1σ uncertainty on D_M(z)/r_d.
        Set to 0 to exclude the D_M observable from this bin.
    """
    z_eff: float
    frac_DH: float
    frac_DM: float

    def __post_init__(self):
        if self.frac_DH < 0 or self.frac_DM < 0:
            raise ValueError("[BAOBin] Fractional uncertainties must be ≥ 0.")
        if self.frac_DH == 0 and self.frac_DM == 0:
            raise ValueError(
                "[BAOBin] At least one of frac_DH, frac_DM must be > 0."
            )


@dataclass
class SurveySpec:
    """Specification for one survey stage.

    Parameters
    ----------
    name : str
        Human-readable label (used in plot titles and result keys).
    rsd_bins : list of FSigma8Bin
        RSD fσ₈ measurement bins.  May be empty.
    bao_bins : list of BAOBin
        BAO D_H/r_d and D_M/r_d measurement bins.  May be empty.
    """
    name: str
    rsd_bins: list = field(default_factory=list)
    bao_bins: list = field(default_factory=list)

    def __post_init__(self):
        if not self.rsd_bins and not self.bao_bins:
            raise ValueError(
                f"[SurveySpec '{self.name}'] Must have at least one measurement bin."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Fisher result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FisherResult:
    """Output of ``FisherForecast.compute()``.

    Attributes
    ----------
    F : (n, n) ndarray
        Fisher information matrix, ordered as in ``param_names``.
    cov : (n, n) ndarray
        Inverse Fisher matrix (parameter covariance).
    sigma : dict
        Marginalized 1σ forecast uncertainties, keyed by parameter name.
    param_names : list of str
        Ordered parameter names matching F rows/columns.
    fiducial : dict
        Fiducial parameter values used for the forecast.
    surveys : list of str
        Names of surveys included.
    """
    F:           np.ndarray
    cov:         np.ndarray
    sigma:       dict
    param_names: list
    fiducial:    dict
    surveys:     list

    def summary(self) -> str:
        """Return a formatted plain-text summary of the forecast."""
        lines = [
            "Fisher Forecast Results",
            "=" * 40,
            f"Surveys : {', '.join(self.surveys)}",
            f"Params  : {', '.join(self.param_names)}",
            "",
            "Fiducial values:",
        ]
        for name, val in self.fiducial.items():
            lines.append(f"  {name:15s} = {val:.6g}")
        lines += ["", "Forecasted 1σ constraints:"]
        for name in self.param_names:
            lines.append(f"  {name:15s}   σ = {self.sigma[name]:.4g}")
        return "\n".join(lines)

    def plot_ellipse(self, param_x: str, param_y: str, ax=None,
                     confidence: float = 1.0, **kwargs):
        """Plot a 2D confidence ellipse in the (param_x, param_y) plane.

        Parameters
        ----------
        param_x, param_y : str
            Parameter names (must be in ``self.param_names``).
        ax : matplotlib Axes, optional
            Target axes.  A new figure is created if None.
        confidence : float
            Number of σ for the ellipse (default 1).
        **kwargs
            Passed to ``matplotlib.patches.Ellipse``.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
        except ImportError:
            raise ImportError(
                "[FisherForecast] matplotlib is required for plot_ellipse."
            )

        ix = self.param_names.index(param_x)
        iy = self.param_names.index(param_y)

        # 2×2 sub-covariance
        C = np.array([
            [self.cov[ix, ix], self.cov[ix, iy]],
            [self.cov[iy, ix], self.cov[iy, iy]],
        ])

        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        w, h  = 2 * confidence * np.sqrt(np.abs(eigvals))

        if ax is None:
            _, ax = plt.subplots()

        kw = dict(fill=False, edgecolor="C0", linewidth=1.5)
        kw.update(kwargs)

        ellipse = Ellipse(
            xy=(self.fiducial[param_x], self.fiducial[param_y]),
            width=w, height=h, angle=angle, **kw
        )
        ax.add_patch(ellipse)
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.autoscale()
        return ax


# ══════════════════════════════════════════════════════════════════════════════
# Main forecast class
# ══════════════════════════════════════════════════════════════════════════════

class FisherForecast:
    """Compute a Fisher-matrix forecast given a COSMIX pipeline and survey specs.

    Parameters
    ----------
    pipeline : Pipeline
        A fully-built and frozen COSMIX Pipeline.  All free parameters of the
        pipeline are included in the forecast.
    fiducial_theta : array-like
        Fiducial parameter vector (length = ``pipeline.pm.ndim``).  Should be
        close to the expected posterior peak to ensure the likelihood is nearly
        Gaussian near the fiducial.
    surveys : sequence of SurveySpec
        One or more survey specifications to include.
    rd_fiducial : float, optional
        Fiducial sound horizon r_d [Mpc] used when converting BAO theory
        predictions DH(z) → DH/r_d.  Overridden by the pipeline's own ``rd``
        value if present.  Default: 147.78 Mpc (eBOSS DR16).
    rel_step : float, optional
        Relative step size for central-difference derivatives.  For parameter
        θ_i the step is ``max(|θ_i| × rel_step, abs_step)``.  Default: 1e-3.
    abs_step : float, optional
        Minimum absolute step size (used when |θ_i| is near zero).
        Default: 1e-5.
    """

    def __init__(
        self,
        pipeline,
        fiducial_theta: Sequence[float],
        surveys: Sequence[SurveySpec],
        rd_fiducial: float | None = None,
        rel_step:    float = 1e-3,
        abs_step:    float = 1e-5,
    ):
        self.pipeline = pipeline
        self.pm       = pipeline.pm
        self.model    = pipeline.model
        self.fiducial = np.asarray(fiducial_theta, dtype=float)
        self.surveys  = list(surveys)
        self.rel_step = rel_step
        self.abs_step = abs_step

        if len(self.fiducial) != self.pm.ndim:
            raise ValueError(
                f"[FisherForecast] fiducial_theta length {len(self.fiducial)} "
                f"does not match pipeline ndim {self.pm.ndim}."
            )

        # Resolve r_d -----------------------------------------------------------
        if rd_fiducial is not None:
            self._rd = float(rd_fiducial)
        elif "rd" in self.pm.free_names:
            idx = self.pm._free_indices["rd"]
            self._rd = float(self.fiducial[idx])
        elif "rd" in self.pm.fixed_names:
            self._rd = float(self.pm._fixed_values["rd"])
        else:
            self._rd = _RD_FIDUCIAL_DEFAULT

        # Build the merged requirements dict once (union of all survey bins)
        self._requirements = self._build_requirements()

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_requirements(self) -> dict:
        """Aggregate theory requirements from all survey bins."""
        req: dict[str, set] = {}
        for survey in self.surveys:
            for b in survey.rsd_bins:
                req.setdefault("fsigma8", set()).add(float(b.z_eff))
            for b in survey.bao_bins:
                if b.frac_DH > 0:
                    req.setdefault("DH", set()).add(float(b.z_eff))
                if b.frac_DM > 0:
                    req.setdefault("DM", set()).add(float(b.z_eff))
        return {k: np.array(sorted(v)) for k, v in req.items()}

    def _eval_observables(self, theta: np.ndarray):
        """Evaluate all observables at theta.

        Returns
        -------
        obs_vec : 1-D array of theory predictions for each observable/bin
        sigma_vec : 1-D array of corresponding 1σ uncertainties (absolute)
        """
        theory = self.model.compute_theory(theta, self._requirements)
        if theory.invalid:
            raise ValueError(
                "[FisherForecast] Theory evaluation failed at the given parameters."
            )

        obs    = []
        sigmas = []

        for survey in self.surveys:
            # RSD: fσ₈ bins
            for b in survey.rsd_bins:
                val = float(theory.eval("fsigma8", np.array([b.z_eff]))[0])
                obs.append(val)
                sigmas.append(b.sigma)

            # BAO bins
            for b in survey.bao_bins:
                if b.frac_DH > 0:
                    dh  = float(theory.eval("DH", np.array([b.z_eff]))[0])
                    rat = dh / self._rd
                    obs.append(rat)
                    sigmas.append(b.frac_DH * rat)   # absolute = fractional × value
                if b.frac_DM > 0:
                    dm  = float(theory.eval("DM", np.array([b.z_eff]))[0])
                    rat = dm / self._rd
                    obs.append(rat)
                    sigmas.append(b.frac_DM * rat)

        return np.array(obs), np.array(sigmas)

    def _step_sizes(self) -> np.ndarray:
        """Per-parameter step sizes for central differences."""
        h = np.empty(self.pm.ndim)
        for name, idx in self.pm._free_indices.items():
            val = float(self.fiducial[idx])
            h[idx] = max(abs(val) * self.rel_step, self.abs_step)
        return h

    def _jacobian(self, obs_fid: np.ndarray) -> np.ndarray:
        """Compute the n_obs × n_params Jacobian via central differences.

        J[k, i] = ∂O_k / ∂θ_i
        """
        n_obs    = len(obs_fid)
        n_params = self.pm.ndim
        J        = np.zeros((n_obs, n_params))
        h        = self._step_sizes()

        for i in range(n_params):
            theta_p = self.fiducial.copy()
            theta_m = self.fiducial.copy()
            theta_p[i] += h[i]
            theta_m[i] -= h[i]

            try:
                obs_p, _ = self._eval_observables(theta_p)
                obs_m, _ = self._eval_observables(theta_m)
            except (ValueError, RuntimeError) as exc:
                param_name = self.pm.free_names[i]
                warnings.warn(
                    f"[FisherForecast] Theory evaluation failed while differentiating "
                    f"'{param_name}' at step ±{h[i]:.2e}: {exc}.  "
                    "Derivative set to zero — check fiducial or increase abs_step.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                J[:, i] = 0.0
                continue

            J[:, i] = (obs_p - obs_m) / (2.0 * h[i])

        return J

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def compute(self) -> FisherResult:
        """Compute and return the Fisher forecast.

        Returns
        -------
        FisherResult
            Contains the Fisher matrix F, covariance matrix (F⁻¹), and
            marginalized 1σ constraints for every free parameter.
        """
        # Evaluate observables and uncertainties at fiducial
        obs_fid, sigmas = self._eval_observables(self.fiducial)

        # Jacobian: shape (n_obs, n_params)
        J = self._jacobian(obs_fid)

        # Fisher matrix: F = J^T diag(1/σ²) J
        inv_var = 1.0 / (sigmas ** 2)
        F = J.T @ (inv_var[:, np.newaxis] * J)   # equivalent to J.T @ diag(1/σ²) @ J

        # Covariance: F^{-1}  (with warning if near-singular)
        try:
            cov = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            warnings.warn(
                "[FisherForecast] Fisher matrix is singular — some parameters "
                "are unconstrained by the survey.  Covariance contains inf/nan.",
                RuntimeWarning,
                stacklevel=2,
            )
            cov = np.full_like(F, np.nan)

        param_names = self.pm.free_names
        fiducial_dict = {
            name: float(self.fiducial[self.pm._free_indices[name]])
            for name in param_names
        }
        sigma_dict = {
            name: float(np.sqrt(np.abs(cov[i, i])))
            for i, name in enumerate(param_names)
        }

        return FisherResult(
            F=F,
            cov=cov,
            sigma=sigma_dict,
            param_names=param_names,
            fiducial=fiducial_dict,
            surveys=[s.name for s in self.surveys],
        )

    def scan_precision_threshold(
        self,
        target_param: str,
        target_sigma: float,
        scale_factor_range: tuple = (0.1, 5.0),
        n_points: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the survey precision scaling needed to achieve a target σ(param).

        All survey uncertainties are multiplied by a common scale factor
        ``α``.  Returns the range of α values and the corresponding σ(param)
        forecasts.  Use this to find the α at which σ(param) = target_sigma
        (the "dangerous precision threshold" of the companion paper).

        Parameters
        ----------
        target_param : str
            Parameter name to track (e.g. ``'lambda0'``).
        target_sigma : float
            Target 1σ constraint value.
        scale_factor_range : (float, float)
            Min and max multiplicative scale factors for survey uncertainties.
        n_points : int
            Number of evaluation points across the range.

        Returns
        -------
        alphas : 1-D array of scale factors
        sigmas : 1-D array of σ(target_param) at each scale factor
        """
        alpha_arr  = np.linspace(scale_factor_range[0], scale_factor_range[1], n_points)
        sigma_arr  = np.empty(n_points)

        original_surveys = self.surveys

        for k, alpha in enumerate(alpha_arr):
            # Build scaled surveys
            scaled = []
            for s in original_surveys:
                scaled.append(SurveySpec(
                    name=s.name,
                    rsd_bins=[
                        FSigma8Bin(b.z_eff, b.sigma * alpha) for b in s.rsd_bins
                    ],
                    bao_bins=[
                        BAOBin(b.z_eff, b.frac_DH * alpha, b.frac_DM * alpha)
                        for b in s.bao_bins
                    ],
                ))
            # Temporarily swap surveys
            self.surveys = scaled
            self._requirements = self._build_requirements()
            try:
                result = self.compute()
                sigma_arr[k] = result.sigma.get(target_param, np.nan)
            except Exception:
                sigma_arr[k] = np.nan

        # Restore original surveys
        self.surveys = original_surveys
        self._requirements = self._build_requirements()

        return alpha_arr, sigma_arr
