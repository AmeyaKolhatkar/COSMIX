"""Implied-A_s diagnostic — per-sample, posterior-propagated.

Answers the question the paper's consistency check poses: *what primordial
amplitude would each posterior sample need in order to reproduce the
sigma80 it prefers?*  Comparing that against Planck's
ln(10^10 A_s) = 3.044 +/- 0.014 is the consistency test.

The relation
------------
The present-day amplitude factorises as

    sigma80^2 = A_s * J(theta) * D0^2 ,                                   (1)

with J the transfer-function/window integral (a function of Omega_m h^2,
Omega_b h^2, n_s, h) and D0 the *total* growth from the transfer epoch to
today.  D0 is not the D(z)/D(0) used for fsigma8 -- that normalisation
divides out precisely the factor needed here.

Running a Boltzmann code for LCDM at a fixed reference amplitude gives
sigma8_ref(theta)^2 = A_s_ref J(theta) D0_LCDM^2, so dividing cancels J and

    ln(10^10 A_s^implied) = ln(10^10 A_s_ref)
                            + 2 ln(sigma80 / sigma8_ref(theta))
                            - 2 ln R(theta) ,   R = D0_model / D0_LCDM     (2)

Both corrections matter, and the second dominates.  Across the revision
posteriors (Revision_Runs/Eg_with_noVIPERS) the transfer term shifts ln A_s by
0.002-0.010, less than the Planck sigma of 0.014, while the growth term
reaches 0.006 (no lambda0) to 0.252 (Hybrid + lambda0, BSDp) -- up to 18
Planck sigma.  Since lambda enters Eq. (1) through D0 alone, dropping R
removes the entire signal the diagnostic exists to detect.

What this module replaces
-------------------------
The previous implementation used the universal scaling

    A_s^implied = A_s_Planck (sigma80 / sigma80_ref)^2

with sigma80_ref taken from one of the runs being plotted.  That is circular
-- the reference lands on Planck by construction rather than by measurement
-- and it drops both J(theta) and D0(lambda).  It is the same defect as the
superseded ``PlanckAsPrior`` likelihood, in figure form.  Eq. (2) has no
reference run and no free normalisation.

Where the physics comes from
----------------------------
Nothing model-specific is re-encoded here.  :class:`EFTDiagnostics` supplies
``f_Q`` and ``Q_of_z`` for the registered model, from which

    E(z) = sqrt(Q(z)/Q(0)) ,        mu_G(z) = 1 / f_Q(Q(z))

follow directly -- exactly the two functions the growth ODE needs.  A model
registered with ``EFTDiagnostics.register_model`` therefore works here with
no further wiring.  :func:`verify_growth_ratio` checks the growth amplitudes
against ``GrowthKinematics.growth_ratio`` so the two cannot drift apart.

Usage
-----
>>> from cosmix.postprocessing.utilities.impiedas import (
...     implied_lnAs_from_run, summarize, plot_implied_lnAs)
>>> res = {lbl: implied_lnAs_from_run(d) for lbl, d in runs.items()}
>>> summarize(res)
>>> plot_implied_lnAs(res, save_fig="implied_As.pdf")

References
----------
- Planck 2018 results VI, A&A 641 A6 (2020); arXiv:1807.06209
"""

import numpy as np
import yaml
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from cosmix.Constants import Omegar0
from cosmix.theory.Solvers_.GrowthSolver import GrowthSolver
from cosmix.postprocessing.EFTDiagnostics import (
    EFTDiagnostics, weighted_quantile,
)

__all__ = [
    "implied_lnAs_from_run", "summarize", "plot_implied_lnAs",
    "growth_ratio_from_samples", "load_sigma8_ref_grid",
    "verify_growth_ratio", "LN_AS_PLANCK", "LN_AS_SIGMA",
]


def as_diagnostic(*args, **kwargs):
    """Removed: the previous implied-A_s diagnostic was circular.

    It computed ``A_s_Planck * (sigma80/sigma80_ref)**2`` with ``sigma80_ref``
    taken from one of the runs being plotted, so that run landed on Planck by
    construction rather than by measurement, and it omitted both the
    transfer-function ratio and the growth factor D0(lambda) (the larger of
    the two).  Results from it are not comparable with the corrected relation
    and must not be mixed with it in one figure or table.

    Migrate to::

        res = {label: implied_lnAs_from_run(path) for label, path in runs.items()}
        summarize(res)
        plot_implied_lnAs(res, save_fig=...)

    Note the reference-run argument is gone: the corrected relation has no
    free normalisation, so there is nothing to anchor.
    """
    raise NotImplementedError(as_diagnostic.__doc__)

#: Planck 2018 TT,TE,EE+lowE+lensing constraint on ln(10^10 A_s).
#: Quoted on the *log*; sigma(A_s) = A_s * LN_AS_SIGMA, not LN_AS_SIGMA*1e-9.
LN_AS_PLANCK = 3.044
LN_AS_SIGMA = 0.014

_DEFAULT_GRID = (Path(__file__).resolve().parents[2]
                 / "nbks_n_others" / "sigma8_ref_grid_wide.npz")

# GrowthSolver integrates a fixed grid N = ln a in [-7, 0] (z ~ 1096 -> 0).
_N_EVAL = np.linspace(-7.0, 0.0, 250)
_Z_EVAL = np.exp(-_N_EVAL) - 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Growth ratio R = D0_model / D0_LCDM
# ══════════════════════════════════════════════════════════════════════════════
class _SampleBG:
    """Background stand-in exposing exactly what ``GrowthSolver.solve`` reads.

    Interpolates pre-computed arrays in N = ln a, so it stays correct if the
    solver's internal grid changes.
    """

    def __init__(self, dlnH_dN, Omegamz, muG):
        self._dlnH, self._Om, self._mu = dlnH_dN, Omegamz, muG

    def z_grid(self):
        return np.array([0.0, 1.0])      # only used for the final interpolation

    def _at(self, z, arr):
        N = -np.log(1.0 + np.asarray(z, dtype=float))
        return np.interp(N, _N_EVAL, arr)

    def dlnH_dN(self, z):
        return self._at(z, self._dlnH)

    def Omegamz(self, z):
        return self._at(z, self._Om)

    def muG(self, z):
        return self._at(z, self._mu)


def _dlnH_dN_from_E(E):
    """d ln H / dN on _N_EVAL, by gradient of ln E (H0 cancels)."""
    return np.gradient(np.log(E), _N_EVAL, axis=-1)


def _D0(dlnH, Om_z, muG):
    """Unnormalised growth amplitude D(a=1); NaN if the integration fails."""
    solver = GrowthSolver(_SampleBG(dlnH, Om_z, muG), None, None)
    try:
        solver.solve()
    except Exception:
        return np.nan
    return solver.unnorm_D0


def growth_ratio_from_samples(diag, Omegam0, weight_floor=1e-10, weights=None,
                              verbose=False):
    """R = D0_model / D0_LCDM for every posterior sample.

    Both amplitudes come from the same ODE, the same N_ini = -7 (z ~ 1096) and
    the same delta ~ a initial condition; the model uses its own E(z) and
    mu_G(z), the reference a LCDM E(z) at the same Omega_m0 with mu_G = 1.
    Everything earlier than N_ini is common to both and cancels, which is what
    licenses pairing R with a sigma8_ref computed by CAMB for LCDM.

    Parameters
    ----------
    diag : EFTDiagnostics
        Supplies the model through ``f_Q`` and ``Q_of_z``: E = sqrt(Q/Q_0) and
        mu_G = 1/f_Q.  A GR model gives mu_G = 1 and hence R = 1 exactly, with
        no special-casing.
    Omegam0 : array_like
        Per-sample matter density (n_sample,).
    weight_floor : float
        Samples carrying less than this fraction of the peak weight are
        skipped and returned as NaN -- they cannot affect any weighted
        summary, and skipping them saves the bulk of the ODE solves.

    Returns
    -------
    ndarray (n_sample,), NaN where not evaluated or where growth diverged.
    """
    Omegam0 = np.asarray(Omegam0, dtype=float)
    n = len(Omegam0)

    Q = diag.Q_of_z(_Z_EVAL)                       # (n, 250)
    Q0 = diag.Q_of_z(np.array([0.0]))              # (n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        E_model = np.sqrt(Q / Q0)
        muG = 1.0 / diag.f_Q(Q)

    OL = 1.0 - Omegam0 - Omegar0
    zp1 = (1.0 + _Z_EVAL)[None, :]
    E_ref = np.sqrt(Omegam0[:, None] * zp1 ** 3
                    + Omegar0 * zp1 ** 4 + OL[:, None])

    dlnH_model = _dlnH_dN_from_E(E_model)
    dlnH_ref = _dlnH_dN_from_E(E_ref)
    Om_model = Omegam0[:, None] * zp1 ** 3 / E_model ** 2
    Om_ref = Omegam0[:, None] * zp1 ** 3 / E_ref ** 2
    ones = np.ones_like(_Z_EVAL)

    if weights is None:
        idx = np.arange(n)
    else:
        w = np.asarray(weights, dtype=float)
        idx = np.flatnonzero(w > w.max() * weight_floor)

    R = np.full(n, np.nan)
    for i in idx:
        if not (np.all(np.isfinite(E_model[i])) and np.all(np.isfinite(muG[i]))):
            continue
        d_ref = _D0(dlnH_ref[i], Om_ref[i], ones)
        d_mod = _D0(dlnH_model[i], Om_model[i], muG[i])
        if np.isfinite(d_ref) and d_ref > 0 and np.isfinite(d_mod) and d_mod > 0:
            R[i] = d_mod / d_ref
    if verbose:
        print(f"  [growth] solved {len(idx)}/{n} samples, "
              f"{int(np.isnan(R[idx]).sum())} failed")
    return R


# ══════════════════════════════════════════════════════════════════════════════
# sigma8_ref and the implied amplitude
# ══════════════════════════════════════════════════════════════════════════════
def load_sigma8_ref_grid(path=None):
    """Load the cached CAMB sigma8_ref table; returns (interp_fn, A_s_ref)."""
    path = Path(path) if path is not None else _DEFAULT_GRID
    if not path.exists():
        raise FileNotFoundError(
            f"sigma8_ref table not found at {path}. Build it with the "
            f"grid-builder script (a few thousand CAMB calls, ~30 min, once)."
        )
    d = np.load(path)
    rgi = RegularGridInterpolator(
        (d["Om_ax"], d["H0_ax"], d["wb_ax"], d["ns_ax"]),
        np.log(d["grid"]), method="linear",
        bounds_error=False, fill_value=np.nan,
    )

    def interp(Omegam0, H0, omega_b, n_s):
        pts = np.column_stack([np.atleast_1d(Omegam0), np.atleast_1d(H0),
                               np.atleast_1d(omega_b), np.atleast_1d(n_s)])
        return np.exp(rgi(pts))

    return interp, float(d["As_ref"])


def implied_lnAs_from_run(run_dir, grid_path=None, weight_floor=1e-10,
                          label=None, verbose=False):
    """Per-sample ln(10^10 A_s^implied) for one archived run, via Eq. (2).

    Returns
    -------
    dict with keys ``lnAs`` (n_sample,), ``weights``, ``R``, ``sigma8_ref``,
    ``sigma80``, ``label``, ``model``, ``run_dir``, ``offgrid_weight``.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "manifest.yaml") as fh:
        manifest = yaml.safe_load(fh)
    names = manifest["labels"]["names"]
    chain = np.load(run_dir / "chain.npy")
    weights = np.load(run_dir / "weights.npy")
    weights = weights / weights.sum()

    def column(key):
        if key in names:
            return chain[:, names.index(key)]
        entry = manifest.get("parameters", {}).get(key) or {}
        if entry.get("value") is None:
            raise KeyError(
                f"{run_dir.name}: {key!r} is neither sampled nor fixed with a "
                f"recorded value; it is required for the implied-A_s relation."
            )
        return np.full(len(chain), float(entry["value"]))

    sigma80 = column("sigma80")
    Omegam0, H0 = column("Omegam0"), column("H0")
    omega_b, n_s = column("omega_b"), column("n_s")

    interp, As_ref = load_sigma8_ref_grid(grid_path)
    sigma8_ref = interp(Omegam0, H0, omega_b, n_s)
    offgrid = float(weights[~np.isfinite(sigma8_ref)].sum())
    if offgrid > 1e-3:
        raise ValueError(
            f"{run_dir.name}: samples outside the sigma8_ref grid carry "
            f"{offgrid:.2e} of the posterior weight (>1e-3). Widen the grid."
        )

    diag = EFTDiagnostics.from_run(run_dir)
    R = growth_ratio_from_samples(diag, Omegam0, weight_floor=weight_floor,
                                  weights=weights, verbose=verbose)

    with np.errstate(divide="ignore", invalid="ignore"):
        lnAs = (np.log(1e10 * As_ref)
                + 2.0 * np.log(sigma80 / sigma8_ref)
                - 2.0 * np.log(R))

    return {
        "lnAs": lnAs, "weights": weights, "R": R, "sigma8_ref": sigma8_ref,
        "sigma80": sigma80, "label": label or run_dir.name,
        "model": manifest["model"]["name"], "run_dir": str(run_dir),
        "offgrid_weight": offgrid,
    }


def summarize(results, quantiles=(0.5, 0.16, 0.84, 0.025, 0.975),
              print_table=True):
    """Weighted quantiles of ln(10^10 A_s) and the tension against Planck.

    The tension is evaluated in **ln A_s**, where the Planck constraint is
    Gaussian; the implied posterior is skewed in linear A_s (A_s ~ sigma80^2),
    so a symmetric +/- there would misrepresent it.
    """
    out = {}
    for label, res in results.items():
        lnAs, w = res["lnAs"], res["weights"]
        ok = np.isfinite(lnAs) & np.isfinite(w)
        q = weighted_quantile(lnAs[ok], w[ok], quantiles)
        med = q[0]
        mean = float(np.average(lnAs[ok], weights=w[ok]))
        sd = float(np.sqrt(np.average((lnAs[ok] - mean) ** 2, weights=w[ok])))
        tension = (mean - LN_AS_PLANCK) / np.sqrt(sd ** 2 + LN_AS_SIGMA ** 2)
        out[label] = {
            "quantiles": list(quantiles), "lnAs_q": q,
            "lnAs_mean": mean, "lnAs_sd": sd, "tension_sigma": float(tension),
            "As_ratio_vs_planck": float(np.exp(mean - LN_AS_PLANCK)),
            "median_R": float(weighted_quantile(res["R"][np.isfinite(res["R"])],
                                                w[np.isfinite(res["R"])], 0.5)),
            "model": res["model"],
        }

    if print_table:
        print(f"{'Model':<34} {'ln(1e10 As)':>18} {'A_s/Planck':>11} "
              f"{'R=D0/D0_LCDM':>13} {'tension':>9}")
        print("-" * 90)
        for label, d in out.items():
            lo, hi = d["lnAs_q"][1], d["lnAs_q"][2]
            print(f"{label:<34} {d['lnAs_mean']:>8.4f} "
                  f"[{lo:.3f},{hi:.3f}] {d['As_ratio_vs_planck']:>11.3f} "
                  f"{d['median_R']:>13.4f} {d['tension_sigma']:>+8.2f}s")
        print(f"\nPlanck 2018: ln(10^10 A_s) = {LN_AS_PLANCK} +/- {LN_AS_SIGMA}"
              f"   (sigma on the log; sigma(A_s) = {LN_AS_SIGMA:.3f} * A_s)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_implied_lnAs(results, save_fig=None, dpi=300, ax=None, colors=None,
                      show_tension=True, in_As_units=False):
    """Forest plot of implied ln(10^10 A_s) against the Planck band.

    A forest plot rather than bars: A_s has no meaningful zero, so bars drawn
    from the origin waste ~60% of the axis and cannot show the posterior
    width.  Here each row carries the weighted median with 68% and 95%
    intervals, against Planck as a vertical band -- the comparison the
    diagnostic is actually making.

    Parameters
    ----------
    in_As_units : bool
        Label the axis as 10^10 A_s instead of ln(10^10 A_s).  The scale
        stays logarithmic in A_s either way, since that is where the Planck
        constraint is Gaussian.
    """
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    n = len(labels)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.2, 0.9 + 0.46 * n))
    cycle = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.axvspan(LN_AS_PLANCK - 2 * LN_AS_SIGMA, LN_AS_PLANCK + 2 * LN_AS_SIGMA,
               color="0.5", alpha=0.18, lw=0, zorder=0)
    ax.axvspan(LN_AS_PLANCK - LN_AS_SIGMA, LN_AS_PLANCK + LN_AS_SIGMA,
               color="0.4", alpha=0.30, lw=0, zorder=1,
               label=r"Planck 2018 $1\sigma$, $2\sigma$")
    ax.axvline(LN_AS_PLANCK, color="k", lw=1.1, zorder=2)

    for i, label in enumerate(labels):
        res = results[label]
        lnAs, w = res["lnAs"], res["weights"]
        ok = np.isfinite(lnAs) & np.isfinite(w)
        med, lo68, hi68, lo95, hi95 = weighted_quantile(
            lnAs[ok], w[ok], [0.5, 0.16, 0.84, 0.025, 0.975])
        y = n - 1 - i
        colour = cycle[i % len(cycle)]
        ax.plot([lo95, hi95], [y, y], color=colour, lw=1.4, alpha=0.55,
                solid_capstyle="butt", zorder=3)
        ax.plot([lo68, hi68], [y, y], color=colour, lw=4.0, alpha=0.9,
                solid_capstyle="butt", zorder=4)
        ax.plot([med], [y], "o", color="white", mec=colour, mew=1.6,
                ms=6, zorder=5)
        if show_tension:
            mean = float(np.average(lnAs[ok], weights=w[ok]))
            sd = float(np.sqrt(np.average((lnAs[ok] - mean) ** 2, weights=w[ok])))
            t = np.abs(mean - LN_AS_PLANCK) / np.sqrt(sd ** 2 + LN_AS_SIGMA ** 2)
            ax.annotate(f"{t:+.1f}$\\sigma$", xy=(1.005, y), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=9, color=colour)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(r"implied $\ln(10^{10}A_s)$", fontsize=12)
    if in_As_units:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{np.exp(t):.1f}" for t in ticks])
        ax.set_xlabel(r"implied $10^{10}A_s$", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    if save_fig is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# Consistency guard
# ══════════════════════════════════════════════════════════════════════════════
def verify_growth_ratio(model_name="fQ_Hybrid", Omegam0=0.31, alpha2=0.72,
                        lambda0=0.6, H0=68.3, rtol=2e-4):
    """Check this module's R against ``GrowthKinematics.growth_ratio``.

    The growth amplitudes here are integrated through a stand-in background
    built from ``EFTDiagnostics``; the engine builds its own from
    ``BackgroundKinematics``.  The two must agree.  Tolerance is loose
    (~1e-4) because the engine interpolates H(z) on a finite grid while this
    module evaluates it analytically -- that difference is real and bounded,
    not a sign of disagreement.

    Raises AssertionError on mismatch; returns the relative deviation.
    """
    from cosmix.core.ParameterManager_ import ParameterManager, Parameter
    from cosmix.core.BackgroundKinematics import BackgroundKinematics
    from cosmix.core.BackgroundConfiguration import BackgroundConfig
    from cosmix.core.GrowthKinematics import GrowthKinematics
    from cosmix.postprocessing.EFTDiagnostics import get_model_spec
    from cosmix.theory.fQ_LCDM import fQLCDM
    from cosmix.theory.fQ_Hybrid import fQHybrid

    cls = {"fQ_LCDM": fQLCDM, "fQ_Hybrid": fQHybrid}[model_name]
    values = {"H0": H0, "Omegam0": Omegam0, "lambda0": lambda0,
              "alpha1": 1.0, "alpha2": alpha2}

    pm = ParameterManager()
    for p in cls.declare_parameters():
        pm.add_parameter(Parameter(name=p.name, latex=p.latex, prior=p.prior,
                                   role=p.role, status="fixed",
                                   value=values.get(p.name, p.value)))
    pm.freeze()
    model, theta = cls(pm), np.array([])

    bg = BackgroundKinematics(model=model, theta=theta,
                              config=BackgroundConfig(z_max=3.0, nz=150,
                                                      z_max_extended=1200.0))
    engine_R = float(np.atleast_1d(
        GrowthKinematics(background=bg, model=model,
                         theta=theta).growth_ratio(np.array([0.0])))[0])

    spec = get_model_spec(model_name)
    params = {k: np.array([values[k]]) for k in spec.params}
    f_Q, f_QQ, Q_of_z = spec.build(params)
    here_R = float(growth_ratio_from_samples(
        EFTDiagnostics(f_Q, f_QQ, Q_of_z), np.array([Omegam0]))[0])

    dev = abs(here_R / engine_R - 1.0)
    assert dev < rtol, (
        f"growth ratio mismatch for {model_name}: this module {here_R:.6f} vs "
        f"GrowthKinematics {engine_R:.6f} (rel dev {dev:.2e})"
    )
    return {"here": here_R, "engine": engine_R, "rel_dev": dev}
