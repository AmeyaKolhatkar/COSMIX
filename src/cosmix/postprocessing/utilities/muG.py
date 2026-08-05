"""Effective gravitational coupling mu_G(z) — plot and derived scalars.

    mu_G(z) = 1 / f_Q(Q(z))                                               (1)

the factor multiplying Newton's constant in the quasi-static growth equation.
mu_G = 1 is the General-Relativity value; mu_G < 1 is weaker gravity and so
suppressed structure growth.

Unlike q and w_eff this needs no differentiation -- ``f_Q`` comes straight
from :class:`EFTDiagnostics`, which is also what makes the module work for
any registered model without further wiring.

On the transition redshift
--------------------------
:func:`muGt` defaults to ``target=1.0``, the crossing back to the GR value,
which is the meaningful "transition" for a coupling. ``mu_G = 0`` is **not**
attainable: it would require f_Q -> infinity, and for the models here mu_G
approaches 1 from below as the modification self-extinguishes at high
redshift. A monotonic approach to 1 also means the crossing may not occur at
finite z at all -- always check the returned fraction before quoting an
interval.

Usage
-----
>>> from cosmix.postprocessing.utilities.muG import muG0, muGt, plot_muG
>>> muG0(run_dir)                      # (median, 16th, 84th) over the posterior
>>> muGt(run_dir, return_fraction=True)
>>> plot_muG({"BD2R": {...}, "BSDp": {...}}, save_fig="mu_G_comparison.pdf")
"""

import numpy as np

from ._kinematics_common import (
    as_diagnostics, muG_of_z, value_at, transition_redshift,
    plot_single_panel, _best_fit_index, Z_ARRAY_LOG, NOTEBOOK_COLORS,
)

__all__ = ["muG_of_z", "muG0", "muGt", "plot_muG"]

#: GR value of the coupling; the reference line in the figure.
MU_GR = 1.0


def muG0(run, best_fit=False, quantiles=(0.5, 0.16, 0.84)):
    """Effective coupling today, mu_G(z=0).

    Returns weighted posterior quantiles, or a scalar at the
    maximum-likelihood point when ``best_fit=True``.
    """
    diag = as_diagnostics(run)
    return value_at(diag, muG_of_z, 0.0, weights=diag.weights,
                    best_fit_index=_best_fit_index(run) if best_fit else None,
                    quantiles=quantiles)


def muGt(run, target=MU_GR, best_fit=False, quantiles=(0.5, 0.16, 0.84),
         z_grid=None, return_fraction=False):
    """Redshift at which mu_G(z) crosses `target` (default: the GR value 1).

    See the module docstring: ``target=0`` is unattainable, and the approach
    to 1 is asymptotic for these models, so the crossing fraction is the
    first thing to check.
    """
    diag = as_diagnostics(run)
    res, frac = transition_redshift(
        diag, muG_of_z, target=target,
        z_grid=Z_ARRAY_LOG if z_grid is None else z_grid,
        weights=diag.weights,
        best_fit_index=_best_fit_index(run) if best_fit else None,
        quantiles=quantiles)
    return (res, frac) if return_fraction else res


def plot_muG(pipelines, z=None, save_fig=None, dpi=1000, best_fit=True,
             colors=None, linestyles=None,
             yticks=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
             x_lims=None, y_lims=None, **kw):
    """mu_G(z) comparison figure — single panel, log-x, GR reference line.

    Layout follows ``PROJECTS_/VIZ_Hybrid_2.ipynb``: the GR line is drawn in
    the LCDM colour, since mu_G = 1 *is* the LCDM curve.
    """
    return plot_single_panel(
        pipelines, muG_of_z, r"$\mu_G(z)$", z=z, colors=colors,
        linestyles=linestyles, yticks=yticks, xscale="log",
        hline=MU_GR, hline_color=NOTEBOOK_COLORS[3],
        save_fig=save_fig, dpi=dpi, best_fit=best_fit, x_lims=x_lims, y_lims=y_lims, **kw)
