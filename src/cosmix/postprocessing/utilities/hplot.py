"""
H(z) plot utility
"""
import numpy as np
import matplotlib.pyplot as plt

def H_plot(datasets, N_band=150, rng_seed=42, save_fig=None, dpi=300,
           figsize=(6, 8), colors=None, linestyles=None, linewidths=None,
           legend_panel=0, reference_label=None):
    r"""
    Stacked H(z) comparison figure with fractional-residual sub-panels.

    Each dataset produces a pair of vertically-joined sub-panels:
      • Top    : absolute H(z)  [km s⁻¹ Mpc⁻¹]
      • Bottom : ΔH/H_ref(z) = [H(z) − H_ref(z)] / H_ref(z)   [%]
    where H_ref is the best-fit curve of `reference_label` (defaults to
    the first model in the dict).

    Parameters
    ----------
    datasets        : dict  {panel_title: labeled_runs_dict}
    N_band          : int   posterior samples for 68 % credible band
    rng_seed        : int
    save_fig        : str or None
    dpi             : int
    figsize         : (width, height)
    colors          : list[str] or None   — one per model
    linestyles      : list[str] or None   — one per model
    linewidths      : list[float] or None — one per model
    legend_panel    : int or "all" or None   — 0-indexed dataset panel index
    reference_label : str or None
        Model label to use as H_ref in the residual panel.
        Defaults to the first model label found.
    """
    import warnings

    z_plot = np.linspace(0.0, 1.8, 400)
    req_plot = {"H": {"z": z_plot}, "DM": {"z": z_plot}, "fsigma8": {"z": z_plot}}

    rng = np.random.default_rng(rng_seed)

    # Build ordered union of model labels
    all_model_labels = []
    seen = set()
    for lrd in datasets.values():
        for lbl in lrd:
            if lbl not in seen:
                all_model_labels.append(lbl)
                seen.add(lbl)
    n_models = len(all_model_labels)

    _ref_label = reference_label if reference_label is not None else all_model_labels[0]

    _default_ls     = ["-", "--", "-.", ":"]
    _default_colors = plt.cm.tab10(np.linspace(0, 0.7, n_models))
    _colors = list(colors)     if colors     is not None else list(_default_colors)
    _ls     = list(linestyles) if linestyles is not None else _default_ls
    _lw     = list(linewidths) if linewidths is not None else [1.5] * n_models

    col_map = {lbl: _colors[i % len(_colors)] for i, lbl in enumerate(all_model_labels)}
    ls_map  = {lbl: _ls[i % len(_ls)]         for i, lbl in enumerate(all_model_labels)}
    lw_map  = {lbl: _lw[i % len(_lw)]         for i, lbl in enumerate(all_model_labels)}

    n_ds = len(datasets)
    # Each dataset gets a (top, bottom) row pair with height ratio 3:1.5
    height_ratios = [h for _ in range(n_ds) for h in (3, 1.5)]
    fig, axes_flat = plt.subplots(
        2 * n_ds, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "height_ratios": height_ratios},
    )
    panel_axes = [(axes_flat[2 * i], axes_flat[2 * i + 1]) for i in range(n_ds)]

    for panel_idx, (panel_title, labeled_runs_dict) in enumerate(datasets.items()):
        ax_top, ax_res = panel_axes[panel_idx]

        # ── compute best-fit H curves for all models ─────────────────────────
        bf_curves = {}
        band_lo   = {}
        band_hi   = {}

        for label, (results, pipeline) in labeled_runs_dict.items():
            theta_bf = results.best_fit
            try:
                theory = pipeline.model.compute_theory(theta_bf, req_plot)
            except Exception as e:
                warnings.warn(f"[H_plot] compute_theory failed for '{label}': {e}")
                continue
            if theory.invalid:
                warnings.warn(f"[H_plot] best-fit theory invalid for '{label}'. Skipping.")
                continue
            bf_curves[label] = theory.eval("H", z_plot)

            chain, wts = results.chain, results.weights
            n_avail = len(chain)
            n_draw  = min(N_band, n_avail)
            if wts is not None:
                w = wts / wts.sum()
                idx_s = rng.choice(n_avail, size=n_draw, replace=False, p=w)
            else:
                idx_s = rng.choice(n_avail, size=n_draw, replace=False)

            samp = []
            for idx in idx_s:
                try:
                    th = pipeline.model.compute_theory(chain[idx], req_plot)
                    if not th.invalid:
                        samp.append(th.eval("H", z_plot))
                except Exception:
                    pass

            if len(samp) >= 10:
                arr = np.array(samp)
                band_lo[label] = np.percentile(arr, 16, axis=0)
                band_hi[label] = np.percentile(arr, 84, axis=0)

        # Reference curve for residual panel
        _ref_local = _ref_label if _ref_label in bf_curves else next(iter(bf_curves))
        H_ref = bf_curves[_ref_local]

        # ── top panel: absolute H(z) ─────────────────────────────────────────
        for label, H_bf in bf_curves.items():
            col = col_map[label]
            ls  = ls_map[label]
            lw  = lw_map[label]
            ax_top.plot(z_plot, H_bf, color=col, ls=ls, lw=lw,
                        label=label, zorder=5)
            if label in band_lo:
                ax_top.fill_between(z_plot, band_lo[label], band_hi[label],
                                    color=col, alpha=0.12, zorder=2)

        ax_top.set_ylabel(r"$H(z)$ ", fontsize=13)
        ax_top.tick_params(labelsize=12)
        ax_top.set_xlim(z_plot[0], z_plot[-1])
        ax_top.text(0.18, 0.95, panel_title,
                    transform=ax_top.transAxes, ha="right", va="top",
                    fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        show_legend = (legend_panel == "all") or (legend_panel == panel_idx)
        if show_legend:
            ax_top.legend(fontsize=11, frameon=False,
                          bbox_to_anchor=(0.5, 1.27), loc="upper center",
                          ncol=3, labelspacing=0.35, columnspacing=4.5)

        # ── bottom panel: fractional residual ΔH/H_ref [%] ───────────────────
        ax_res.axhline(0, color=col_map.get(_ref_local, "k"),
                       lw=0.8, ls="-", zorder=4)
        ax_res.axhspan(-0.5, 0.5, color="gray", alpha=0.08, zorder=1)

        for label, H_bf in bf_curves.items():
            if label == _ref_local:
                continue
            col = col_map[label]
            ls  = ls_map[label]
            lw  = lw_map[label]
            resid = (H_bf - H_ref) / H_ref * 100.0
            ax_res.plot(z_plot, resid, color=col, ls=ls, lw=lw, zorder=5)
            if label in band_lo and label in band_hi:
                resid_lo = (band_lo[label] - H_ref) / H_ref * 100.0
                resid_hi = (band_hi[label] - H_ref) / H_ref * 100.0
                ax_res.fill_between(z_plot, resid_lo, resid_hi,
                                    color=col, alpha=0.15, zorder=2)

        ax_res.set_ylabel(r"$\Delta H/H_{\rm ref}$  [%]", fontsize=11)
        ax_res.tick_params(labelsize=12)
        ax_res.yaxis.set_major_locator(plt.MaxNLocator(3, symmetric=True))

        ax_top.spines["bottom"].set_linewidth(0.6)
        ax_res.spines["top"].set_linewidth(0.6)

    axes_flat[-1].set_xlabel(r"Redshift $z$", fontsize=13)
    axes_flat[-1].set_xticks([0.0, 0.5, 1.0, 1.5])

    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    plt.show()