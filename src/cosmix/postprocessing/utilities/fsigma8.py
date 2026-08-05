"""
fsigma8 plot utility
"""
import numpy as np
import matplotlib.pyplot as plt


def fsigma8_plot(datasets, N_band=150, rng_seed=42, save_fig=None, dpi=300, figsize=(8, 9),
                 colors=None, linestyles=None, linewidths=None, legend_panel=0):
    r"""
    Stacked f*sigma8(z) comparison figure.

    Parameters
    ----------
    datasets : dict  {panel_title: labeled_runs_dict}
        Each entry produces one panel, stacked vertically.
        labeled_runs_dict : {model_label: (MCMCResults, Pipeline)}
    N_band      : int    number of posterior samples for 68% credible band
    rng_seed    : int
    save_fig    : str or None
    figsize     : (width, height) — total figure size
    colors      : list[str] or None
        Custom colours, one per model in the order they appear across all
        datasets.  Falls back to tab10 if None.
    linestyles  : list[str] or None
        Custom line styles (e.g. ["-", "--", "-.", ":"]), one per model.
        Falls back to the default cycle if None.
    linewidths  : list[float] or None
        Custom line widths, one per model. Falls back to 1.5 if None.
    legend_panel : int or "all" or None
        Which panel (0-indexed) shows the legend.
        Pass "all" to show a legend on every panel, None to suppress entirely.
    """
    import warnings

    z_plot = np.linspace(0.0, 1.8, 400)
    req_plot = {
        "fsigma8": {"z": z_plot},
        "H":       {"z": z_plot},
        "DM":      {"z": z_plot},
    }

    rng = np.random.default_rng(rng_seed)

    # Build ordered union of model labels across all datasets
    all_model_labels = []
    seen = set()
    for lrd in datasets.values():
        for lbl in lrd:
            if lbl not in seen:
                all_model_labels.append(lbl)
                seen.add(lbl)
    n_models = len(all_model_labels)

    _default_ls = ["-", "--", "-.", ":"]
    _default_colors = plt.cm.tab10(np.linspace(0, 0.7, n_models))

    _colors = list(colors) if colors is not None else list(_default_colors)
    _ls     = list(linestyles) if linestyles is not None else _default_ls
    _lw     = list(linewidths) if linewidths is not None else [1.5] * n_models

    col_map = {lbl: _colors[i % len(_colors)] for i, lbl in enumerate(all_model_labels)}
    ls_map  = {lbl: _ls[i % len(_ls)]         for i, lbl in enumerate(all_model_labels)}
    lw_map  = {lbl: _lw[i % len(_lw)]         for i, lbl in enumerate(all_model_labels)}

    n_panels = len(datasets)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=figsize,
                             sharex=True,
                             gridspec_kw={"hspace": 0.08})
    if n_panels == 1:
        axes = [axes]

    for panel_idx, (panel_title, labeled_runs_dict) in enumerate(datasets.items()):
        ax = axes[panel_idx]

        # ── data points ──────────────────────────────────────────────────────
        data_plotted = False
        for _lbl, (_res, pipe) in labeled_runs_dict.items():
            if data_plotted:
                break
            for lik in pipe.likelihoods:
                if hasattr(lik, "fs8") and hasattr(lik, "z"):
                    ax.errorbar(lik.z, lik.fs8, yerr=lik.fs8_err,
                                fmt="o", color="k", mfc="#D6860F", mec="k",
                                ms=5, ecolor="gray", elinewidth=1.0,
                                capsize=3, zorder=10, label=r"$f\sigma_8$ DATA")
                    data_plotted = True
                    break
                if hasattr(lik, "_gauss_lbl") and hasattr(lik, "_gauss_z"):
                    mask = np.array([l == "f_sigma8" for l in lik._gauss_lbl])
                    if mask.any():
                        sigma = np.sqrt(np.diag(lik._gauss_cov))[mask]
                        ax.errorbar(lik._gauss_z[mask],
                                    lik._gauss_val[mask],
                                    yerr=sigma,
                                    fmt="o", color="k", mfc="#D6860F", mec="k",
                                    ms=5, ecolor="gray", elinewidth=1.0,
                                    capsize=3, zorder=10, label=r"$f\sigma_8$ DATA")
                        data_plotted = True
                        break

        # ── model curves + credible bands ────────────────────────────────────
        for label, (results, pipeline) in labeled_runs_dict.items():
            col = col_map[label]
            ls  = ls_map[label]
            lw  = lw_map[label]

            theta_bf = results.best_fit
            try:
                theory = pipeline.model.compute_theory(theta_bf, req_plot)
            except Exception as e:
                warnings.warn(f"[fsigma8_plot] compute_theory failed for '{label}': {e}")
                continue
            if theory.invalid:
                warnings.warn(f"[fsigma8_plot] best-fit theory invalid for '{label}'. Skipping.")
                continue

            fs8_bf = theory.eval("fsigma8", z_plot)
            ax.plot(z_plot, fs8_bf, color=col, ls=ls, lw=lw, label=label, zorder=5)

            # 68% credible band
            chain, wts = results.chain, results.weights
            n_avail = len(chain)
            n_draw  = min(N_band, n_avail)
            if wts is not None:
                w = wts / wts.sum()
                idx_s = rng.choice(n_avail, size=n_draw, replace=False, p=w)
            else:
                idx_s = rng.choice(n_avail, size=n_draw, replace=False)

            band_curves = []
            for idx in idx_s:
                try:
                    th = pipeline.model.compute_theory(chain[idx], req_plot)
                    if not th.invalid:
                        band_curves.append(th.eval("fsigma8", z_plot))
                except Exception:
                    pass

            if len(band_curves) >= 10:
                band = np.array(band_curves)
                ax.fill_between(z_plot,
                                np.percentile(band, 16, axis=0),
                                np.percentile(band, 84, axis=0),
                                color=col, alpha=0.15, zorder=2)

        ax.set_ylabel(r"$f\sigma_8(z)$", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.set_xlim(z_plot[0], z_plot[-1])
        ax.set_yticks([0.25, 0.35, 0.45, 0.55, 0.65])
        ax.set_ylim(0.2, 0.7)

        # Panel label in upper-right corner
        ax.text(0.97, 0.95, panel_title,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=13, 
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        # Legend placement logic
        show_legend = (legend_panel == "all") or (legend_panel == panel_idx)
        if show_legend:
            ax.legend(fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.12), loc="center",
                      ncol=3, labelspacing=0.3)

    axes[-1].set_xlabel(r"Redshift $z$", fontsize=15)
    axes[-1].set_xticks([0.0, 0.5, 1.0, 1.5])

    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    plt.show()