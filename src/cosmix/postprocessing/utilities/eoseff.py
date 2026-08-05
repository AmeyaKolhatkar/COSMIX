"""Effective equation of state w_eff(z) — plot and derived scalars.

    w_eff(z) = -1 - (2/3) d ln H / dN ,        N = ln a                   (1)

Related to the deceleration parameter by

    q = (1 + 3 w_eff) / 2 ,                                               (2)

an identity, not an approximation.  Two consequences worth noting:

* The accelerated-expansion threshold is ``w_eff = -1/3``, *not* ``w_eff = 0``
  -- by Eq. (2) it is exactly where q = 0.  :func:`wefft` therefore defaults
  to ``target=-1/3``, and should return the same redshift as
  :func:`~deceleration.qt`.  That agreement is a useful cross-check on both.
* ``w_eff = 0`` is not a transition of interest here: w_eff approaches 0 only
  asymptotically deep in matter domination, so it has no root at finite z in
  these models.  Pass ``target=0.0`` explicitly if you want it anyway.

The background comes from :class:`EFTDiagnostics`, so any registered model
works with no further wiring; see :mod:`_kinematics_common`.

Usage
-----
>>> from cosmix.postprocessing.utilities.eoseff import weff0, wefft, plot_weff
>>> weff0(run_dir)                    # (median, 16th, 84th) over the posterior
>>> wefft(run_dir)                    # acceleration threshold, w_eff = -1/3
>>> plot_weff({"BD2R": {...}, "BSDp": {...}}, save_fig="weff_comparison.pdf")
"""

import numpy as np

from ._kinematics_common import (
    as_diagnostics, weff_of_z, value_at, transition_redshift,
    plot_three_panel, _best_fit_index, Z_ARRAY,
)

__all__ = ["weff_of_z", "weff0", "wefft", "plot_weff", "W_ACCEL"]

#: Accelerated-expansion threshold; equivalent to q = 0 by Eq. (2).
W_ACCEL = -1.0 / 3.0


def weff0(run, best_fit=False, quantiles=(0.5, 0.16, 0.84)):
    """Effective equation of state today, w_eff(z=0).

    Returns weighted posterior quantiles, or a scalar at the
    maximum-likelihood point when ``best_fit=True``.
    """
    diag = as_diagnostics(run)
    return value_at(diag, weff_of_z, 0.0, weights=diag.weights,
                    best_fit_index=_best_fit_index(run) if best_fit else None,
                    quantiles=quantiles)


def wefft(run, target=W_ACCEL, best_fit=False, quantiles=(0.5, 0.16, 0.84),
          z_grid=None, return_fraction=False):
    """Transition redshift: the root of w_eff(z) = `target`.

    Defaults to the acceleration threshold ``-1/3``, which by q = (1+3w)/2 is
    the same redshift :func:`~deceleration.qt` returns.  See the module
    docstring on why ``target=0`` has no finite root.
    """
    diag = as_diagnostics(run)
    res, frac = transition_redshift(
        diag, weff_of_z, target=target,
        z_grid=Z_ARRAY if z_grid is None else z_grid,
        weights=diag.weights,
        best_fit_index=_best_fit_index(run) if best_fit else None,
        quantiles=quantiles)
    return (res, frac) if return_fraction else res


def plot_weff(pipelines, z=None, save_fig=None, dpi=1000, best_fit=True,
              colors=None, linestyles=None,
              yticks=(-1.0, -0.8, -0.6, -0.4, -0.2, 0.0),
              zoom1=None, zoom2=None, **kw):
    """w_eff(z) comparison figure — main panel plus two zooms.

    Layout matches :func:`~deceleration.plot_q`; the zoom windows default to
    the acceleration threshold and the de Sitter approach.
    """
    zoom1 = {"xlim": (0.57, 0.7), "ylim": (W_ACCEL - 0.03, W_ACCEL + 0.03),
             "ylabel": r"$transition$", "hline": W_ACCEL,
             "xticks": [0.58, 0.62, 0.66, 0.7]} if zoom1 is None else zoom1
    zoom2 = {"xlim": (-1, -0.8), "ylim": (-1.0005, -0.994),
             "ylabel": r"$future$",
             "yticks": [-1.000, -0.9975, -0.995],
             "xticks": [-1.0, -0.95, -0.9, -0.85, -0.8]} if zoom2 is None else zoom2

    return plot_three_panel(
        pipelines, weff_of_z, r"$\omega_{eff}(z)$", z=z, colors=colors,
        linestyles=linestyles, yticks=yticks, vline=0.0,
        zoom1=zoom1, zoom2=zoom2, save_fig=save_fig, dpi=dpi,
        best_fit=best_fit, **kw)
