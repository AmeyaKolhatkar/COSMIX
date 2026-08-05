"""Deceleration parameter q(z) — plot and derived scalars.

    q(z) = -1 - d ln H / dN ,        N = ln a                             (1)

q < 0 is accelerated expansion, so the root q(z_t) = 0 is the
deceleration-to-acceleration **transition redshift** -- the quantity
:func:`qt` returns.

The background comes from :class:`EFTDiagnostics`, so any model registered
with ``EFTDiagnostics.register_model`` works here with no further wiring; see
:mod:`_kinematics_common`.

Figure layout reproduces ``PROJECTS_/VIZ_Hybrid_2.ipynb``: a main panel over
z in [-1, 3] with a zoom on the transition and a zoom on the de Sitter
future limit, colour encoding the model and line style the pipeline.

Usage
-----
>>> from cosmix.postprocessing.utilities.deceleration import q0, qt, plot_q
>>> q0(run_dir)                       # (median, 16th, 84th) over the posterior
>>> q0(run_dir, best_fit=True)        # scalar at the MAP point
>>> qt(run_dir)                       # transition redshift, same conventions
>>> plot_q({"BD2R": {...}, "BSDp": {...}}, save_fig="q_comparison.pdf")
"""

import numpy as np

from ._kinematics_common import (
    as_diagnostics, q_of_z, value_at, transition_redshift, plot_three_panel,
    _best_fit_index, Z_ARRAY,
)

__all__ = ["q_of_z", "q0", "qt", "plot_q"]


def q0(run, best_fit=False, quantiles=(0.5, 0.16, 0.84)):
    """Deceleration parameter today, q(z=0).

    Returns weighted posterior quantiles, or a scalar at the
    maximum-likelihood point when ``best_fit=True``.
    """
    diag = as_diagnostics(run)
    return value_at(diag, q_of_z, 0.0, weights=diag.weights,
                    best_fit_index=_best_fit_index(run) if best_fit else None,
                    quantiles=quantiles)


def qt(run, best_fit=False, quantiles=(0.5, 0.16, 0.84), z_grid=None,
       return_fraction=False):
    """Transition redshift: the root of q(z) = 0.

    Samples with no crossing inside `z_grid` are dropped; inspect the
    crossing fraction (``return_fraction=True``) before quoting an interval
    from a posterior that only partly transitions in range.
    """
    diag = as_diagnostics(run)
    res, frac = transition_redshift(
        diag, q_of_z, target=0.0,
        z_grid=Z_ARRAY if z_grid is None else z_grid,
        weights=diag.weights,
        best_fit_index=_best_fit_index(run) if best_fit else None,
        quantiles=quantiles)
    return (res, frac) if return_fraction else res


def plot_q(pipelines, z=None, save_fig=None, dpi=1000, best_fit=True,
           colors=None, linestyles=None,
           yticks=(-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4),
           zoom1=None, zoom2=None, **kw):
    """q(z) comparison figure — main panel plus transition and future zooms.

    Parameters
    ----------
    pipelines : dict
        ``{pipeline_label: {model_label: run}}``; `run` is a run directory or
        an :class:`EFTDiagnostics`.  Colour encodes model, style pipeline.
    zoom1, zoom2 : dict, optional
        Axis settings for the zoom panels.  Defaults reproduce the notebook:
        the transition window and the de Sitter approach.  Pass None to keep
        the defaults, or a dict to override.
    """
    zoom1 = {"xlim": (0.57, 0.7), "ylim": (-0.04, 0.04),
             "ylabel": r"$transition$", "hline": 0.0,
             "yticks": [-0.04, -0.02, 0.0, 0.02, 0.04],
             "xticks": [0.58, 0.62, 0.66, 0.7]} if zoom1 is None else zoom1
    zoom2 = {"xlim": (-1, -0.8), "ylim": (-1.0005, -0.994),
             "ylabel": r"$future$",
             "yticks": [-1.000, -0.9975, -0.995],
             "xticks": [-1.0, -0.95, -0.9, -0.85, -0.8]} if zoom2 is None else zoom2

    return plot_three_panel(
        pipelines, q_of_z, r"$q(z)$", z=z, colors=colors,
        linestyles=linestyles, yticks=yticks, hline=0.0,
        zoom1=zoom1, zoom2=zoom2, save_fig=save_fig, dpi=dpi,
        best_fit=best_fit, **kw)
