"""Shared internals for the background-kinematics plot utilities.

Private support for :mod:`deceleration`, :mod:`eoseff` and :mod:`muG`.  Holds
the pieces those three have in common: posterior/best-fit evaluation of a
scalar derived quantity, transition-redshift root finding, and the two figure
skeletons used in ``PROJECTS_/VIZ_Hybrid_2.ipynb`` (a three-panel main+zoom
layout, and a single-panel log-x layout).

All background information is taken from :class:`EFTDiagnostics`, which
supplies ``f_Q`` and ``Q_of_z`` for any registered model.  From those,

    E(z) = sqrt(Q(z)/Q(0)) ,        mu_G(z) = 1 / f_Q(Q(z))

so nothing model-specific is re-encoded here and a model registered with
``EFTDiagnostics.register_model`` works throughout.

The kinematic quantities follow from a single derivative:

    q(z)     = -1 - dlnH/dN
    w_eff(z) = -1 - (2/3) dlnH/dN                                        (1)

whence q = (1 + 3 w_eff)/2 identically -- so q = 0 and w_eff = -1/3 mark the
same redshift, which :func:`transition_redshift` exploits as a cross-check.
"""

import numpy as np
from pathlib import Path

from cosmix.postprocessing.EFTDiagnostics import (
    EFTDiagnostics, weighted_quantile,
)

# ── Palette and line styles from VIZ_Hybrid_2.ipynb ──────────────────────────
#: Model colours, in the notebook's order.  Assigned to labels by insertion
#: order unless an explicit mapping is supplied.
NOTEBOOK_COLORS = ["#808080", "#2ca087", "#DF5595",
                   "#0A89DD", "#C2B540", "#c57a3d"]

#: Line style per pipeline, in insertion order (BD2R solid, BSDp dashed).
NOTEBOOK_LINESTYLES = ["-", "--", "-.", ":"]

#: Legend font used throughout the notebook figures.
LEGEND_FONT = {"size": 11, "family": "Cambria"}

#: Default z grid for the linear-axis figures (notebook: ``z_array``).
Z_ARRAY = np.linspace(-1.0, 3.0, 1000)

#: Notebook's ``z_array2``.  Retained for root-finding, where its coverage of
#: z < 0 matters and the spacing does not.
Z_ARRAY_LOG = np.linspace(-1.0, 1100.0, 1000)

#: Grid for *plotting* on a log x-axis.  ``Z_ARRAY_LOG`` is linear with a
#: spacing of ~1.1, so it places only a couple of points below z = 1 and
#: renders the low-z end -- where the modification is largest -- as visible
#: straight segments; its z < 0 half cannot appear on a log axis at all.
#: Log spacing puts the resolution where the curve actually varies.
Z_ARRAY_LOGPLOT = np.logspace(-2.0, np.log10(1100.0), 800)


# ══════════════════════════════════════════════════════════════════════════════
# Background derivatives
# ══════════════════════════════════════════════════════════════════════════════
def as_diagnostics(run, **kw):
    """Accept a run directory, or an already-built EFTDiagnostics."""
    if isinstance(run, EFTDiagnostics):
        return run
    return EFTDiagnostics.from_run(Path(run), **kw)


def E_of_z(diag, z):
    """E(z) = H/H0 on an (n_sample, n_z) grid, from Q ∝ H^2."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(diag.Q_of_z(z) / diag.Q_of_z(np.array([0.0])))


def dlnH_dN(diag, z, h=1e-5):
    """d ln H / dN with N = ln a, by central difference in z.

    Uses ``dlnH/dN = -(1+z) dlnE/dz`` rather than differencing directly in
    ln a.  The two are equivalent, but the ln a route is degenerate at
    z = -1 (where a -> infinity and the perturbed redshifts collapse onto
    each other), while this form stays well defined and correctly returns 0
    there -- the de Sitter limit q = -1 that the notebook's ``future`` zoom
    panel is built to display.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        lnE_p = np.log(E_of_z(diag, z + h))
        lnE_m = np.log(E_of_z(diag, z - h))
        dlnE_dz = (lnE_p - lnE_m) / (2.0 * h)
    return -(1.0 + z)[None, :] * dlnE_dz


def q_of_z(diag, z, h=1e-5):
    """Deceleration parameter, Eq. (1)."""
    return -1.0 - dlnH_dN(diag, z, h)


def weff_of_z(diag, z, h=1e-5):
    """Effective equation of state, Eq. (1)."""
    return -1.0 - (2.0 / 3.0) * dlnH_dN(diag, z, h)


def muG_of_z(diag, z):
    """mu_G(z) = 1 / f_Q -- exact, no differentiation."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / diag.f_Q(diag.Q_of_z(z))


# ══════════════════════════════════════════════════════════════════════════════
# Scalar summaries
# ══════════════════════════════════════════════════════════════════════════════
def _best_fit_index(run_dir):
    """Index of the maximum-likelihood dead point."""
    return int(np.argmax(np.load(Path(run_dir) / "log_prob.npy")))


def value_at(diag, func, z0=0.0, weights=None, best_fit_index=None,
             quantiles=(0.5, 0.16, 0.84)):
    """Evaluate `func` at a single redshift, as a scalar or a posterior.

    Returns the best-fit scalar when `best_fit_index` is given, otherwise
    weighted quantiles across the posterior.
    """
    vals = np.atleast_2d(func(diag, np.array([float(z0)])))[:, 0]
    if best_fit_index is not None:
        return float(vals[best_fit_index])
    ok = np.isfinite(vals)
    w = None if weights is None else np.asarray(weights)[ok]
    return weighted_quantile(vals[ok], w, quantiles)


def transition_redshift(diag, func, target=0.0, z_grid=None, weights=None,
                        best_fit_index=None, quantiles=(0.5, 0.16, 0.84),
                        min_fraction=0.5):
    """Redshift at which `func` crosses `target`.

    Scans `z_grid` for the first sign change of ``func - target`` and refines
    it by linear interpolation between the bracketing nodes.  Samples with no
    crossing in range yield NaN and are dropped from the weighted summary.

    A summary built from a small minority of samples is not a transition
    redshift -- it is an artefact of wherever those few crossings happen to
    fall.  So when less than `min_fraction` of the posterior weight crosses,
    a ``RuntimeWarning`` is issued naming the fraction, and if nothing
    crosses at all the result is NaN rather than a fabricated number.  This
    is the expected outcome for a quantity that approaches `target`
    asymptotically (mu_G -> 1, for instance) rather than crossing it.

    Returns
    -------
    (result, fraction)
        `result` is a scalar (best-fit) or weighted quantiles; `fraction` is
        the posterior weight fraction that actually crosses `target`.
    """
    import warnings
    z_grid = Z_ARRAY if z_grid is None else np.asarray(z_grid, dtype=float)
    vals = np.atleast_2d(func(diag, z_grid)) - target      # (n_sample, n_z)

    n = vals.shape[0]
    z_t = np.full(n, np.nan)
    sign = np.signbit(vals)
    changes = sign[:, :-1] != sign[:, 1:]
    finite = np.isfinite(vals[:, :-1]) & np.isfinite(vals[:, 1:])
    changes &= finite

    rows, cols = np.nonzero(changes)
    seen = np.zeros(n, dtype=bool)
    for r, c in zip(rows, cols):
        if seen[r]:
            continue                      # keep the first crossing only
        seen[r] = True
        v0, v1 = vals[r, c], vals[r, c + 1]
        z0, z1 = z_grid[c], z_grid[c + 1]
        z_t[r] = z0 - v0 * (z1 - z0) / (v1 - v0)   # linear refinement

    ok = np.isfinite(z_t)
    if weights is None:
        frac = float(ok.mean())
        w_ok = None
    else:
        w = np.asarray(weights, dtype=float)
        frac = float(w[ok].sum() / w.sum())
        w_ok = w[ok]

    if frac < min_fraction:
        warnings.warn(
            f"only {frac:.1%} of the posterior weight crosses target="
            f"{target:g} within z in [{z_grid[0]:g}, {z_grid[-1]:g}]; the "
            f"returned value summarises that minority alone and is not a "
            f"transition redshift. If the quantity approaches the target "
            f"asymptotically rather than crossing it, no such redshift "
            f"exists.",
            RuntimeWarning, stacklevel=3,
        )

    if best_fit_index is not None:
        return float(z_t[best_fit_index]), frac
    if not ok.any():
        return np.full(np.shape(quantiles), np.nan), frac
    return weighted_quantile(z_t[ok], w_ok, quantiles), frac


# ══════════════════════════════════════════════════════════════════════════════
# Figure skeletons
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_style(pipelines, colors, linestyles):
    """Map model labels -> colour and pipeline labels -> line style."""
    model_labels = []
    for runs in pipelines.values():
        for lbl in runs:
            if lbl not in model_labels:
                model_labels.append(lbl)
    if colors is None:
        colors = {lbl: NOTEBOOK_COLORS[i % len(NOTEBOOK_COLORS)]
                  for i, lbl in enumerate(model_labels)}
    if linestyles is None:
        linestyles = {p: NOTEBOOK_LINESTYLES[i % len(NOTEBOOK_LINESTYLES)]
                      for i, p in enumerate(pipelines)}
    return model_labels, colors, linestyles


def _legend_pair(ax, model_labels, colors, linestyles, model_bbox, ls_bbox,
                 model_loc="center right"):
    """The notebook's two-group legend: colour = model, style = pipeline."""
    from matplotlib.lines import Line2D

    model_handles = [Line2D([], [], color=colors[l], lw=1.5, ls="-", label=l)
                     for l in model_labels]
    ls_handles = [Line2D([], [], color="k", lw=3, ls=s, label=p)
                  for p, s in linestyles.items()]

    leg1 = ax.legend(handles=model_handles, bbox_to_anchor=model_bbox,
                     loc=model_loc, prop=LEGEND_FONT, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=ls_handles, bbox_to_anchor=ls_bbox, loc="lower right",
              ncol=2, prop=LEGEND_FONT, frameon=False)


def plot_three_panel(pipelines, func, ylabel, *, z=None, colors=None,
                     linestyles=None, yticks=None, xticks=(-1.0, 0.0, 1.0, 2.0, 3.0),
                     xlim=(-1, 3), hline=None, vline=None,
                     zoom1=None, zoom2=None, model_bbox=(0.98, 0.28),
                     ls_bbox=(0.98, 0.02), figsize=(5, 10),
                     save_fig=None, dpi=1000, best_fit=True):
    """Main + two zoom panels, as in the notebook's q(z) and w_eff(z) figures.

    `pipelines` maps a pipeline label to ``{model_label: run}``; colour
    encodes the model, line style the pipeline.  Each `run` may be a run
    directory or an :class:`EFTDiagnostics`.

    `zoom1` / `zoom2` are dicts of axis settings (``xlim``, ``ylim``,
    ``ylabel``, ``xticks``, ``yticks``, ``hline``); pass None to leave a
    panel blank.
    """
    import matplotlib.pyplot as plt

    z = Z_ARRAY if z is None else np.asarray(z, dtype=float)
    model_labels, colors, linestyles = _resolve_style(pipelines, colors, linestyles)

    fig, (ax_main, ax_zoom1, ax_zoom2) = plt.subplots(
        3, 1, figsize=figsize, sharex=False,
        gridspec_kw={"height_ratios": [4, 2, 2]})

    curves = {}
    for pipe, runs in pipelines.items():
        for lbl, run in runs.items():
            diag = as_diagnostics(run)
            vals = np.atleast_2d(func(diag, z))
            idx = (_best_fit_index(run)
                   if best_fit and not isinstance(run, EFTDiagnostics) else 0)
            curves[(pipe, lbl)] = vals[idx]

    for ax in (ax_main, ax_zoom1, ax_zoom2):
        for (pipe, lbl), y in curves.items():
            ax.plot(z, y, color=colors[lbl], lw=1.5, ls=linestyles[pipe])

    if hline is not None:
        ax_main.axhline(hline, ls="-", lw=1, color="k", alpha=0.6)
    if vline is not None:
        ax_main.axvline(vline, ls="-", lw=1, color="k", alpha=0.6)
    ax_main.set_ylabel(ylabel, fontsize=15)
    if yticks is not None:
        ax_main.set_yticks(yticks)
    ax_main.tick_params(axis="y", labelsize=13)
    ax_main.set_xticks(list(xticks))
    ax_main.tick_params(axis="x", labelsize=12)
    ax_main.set_xlim(*xlim)

    _legend_pair(ax_main, model_labels, colors, linestyles, model_bbox, ls_bbox)

    for ax, spec, is_last in ((ax_zoom1, zoom1, False), (ax_zoom2, zoom2, True)):
        if spec is None:
            ax.set_visible(False)
            continue
        if spec.get("hline") is not None:
            ax.axhline(spec["hline"], ls="-", lw=1, color="k", alpha=0.6)
        if "ylim" in spec:
            ax.set_ylim(*spec["ylim"])
        if "xlim" in spec:
            ax.set_xlim(*spec["xlim"])
        if "ylabel" in spec:
            ax.set_ylabel(spec["ylabel"], fontsize=15)
        if spec.get("yticks") is not None:
            ax.set_yticks(spec["yticks"])
        if spec.get("xticks") is not None:
            ax.set_xticks(spec["xticks"])
        ax.tick_params(axis="y", labelsize=13)
        ax.tick_params(axis="x", labelsize=12)
        if is_last:
            ax.set_xlabel(r"$z$", fontsize=15)

    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    return fig, (ax_main, ax_zoom1, ax_zoom2)


def plot_single_panel(pipelines, func, ylabel, *, z=None, colors=None,
                      linestyles=None, yticks=None, xscale="log",
                      hline=None, hline_color=None, figsize=(7, 5),
                      model_bbox=(0.97, 0.07), ls_bbox=(0.98, 0.02),
                      save_fig=None, dpi=1000, best_fit=True, x_lims=None, y_lims=None):
    """Single panel with log-x, as in the notebook's mu_G(z) figure.
    plot_lims: (xlim_low, xlim_high, ylim_low, ylim_high)"""
    import matplotlib.pyplot as plt

    z = Z_ARRAY_LOGPLOT if z is None else np.asarray(z, dtype=float)
    model_labels, colors, linestyles = _resolve_style(pipelines, colors, linestyles)

    fig, ax = plt.subplots(figsize=figsize)
    for pipe, runs in pipelines.items():
        for lbl, run in runs.items():
            diag = as_diagnostics(run)
            vals = np.atleast_2d(func(diag, z))
            idx = (_best_fit_index(run)
                   if best_fit and not isinstance(run, EFTDiagnostics) else 0)
            ax.plot(z, vals[idx], color=colors[lbl], lw=1.5, ls=linestyles[pipe])

    if hline is not None:
        ax.axhline(hline, ls="-", lw=1.5,
                   color=hline_color or "k", alpha=0.6)
    ax.set_ylabel(ylabel, fontsize=15)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xscale(xscale)
    ax.set_xlabel(r"$z$", fontsize=15)
    if x_lims is not None:
        ax.set_xlim(x_lims[0], x_lims[1])
    if y_lims is not None:
        ax.set_ylim(y_lims[0], y_lims[1])

    _legend_pair(ax, model_labels, colors, linestyles, model_bbox, ls_bbox,
                 model_loc="lower right")

    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    return fig, ax
