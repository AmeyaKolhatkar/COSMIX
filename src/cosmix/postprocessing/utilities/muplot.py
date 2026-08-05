"""
mu (distance modulus) plot utility
"""
import numpy as np
import matplotlib.pyplot as plt

def mu_plot(datasets, N_band=150, rng_seed=42, save_fig=None, dpi=300,
           figsize=(6, 8), colors=None, linestyles=None, linewidths=None,
           legend_panel=0, reference_label=None):
    r"""
    Stacked Hubble diagram with absolute-residual sub-panels.

    Each dataset produces a pair of vertically-joined sub-panels:
      • Top    : distance modulus μ(z)  [mag],  data shown as m_obs − M_MAP
      • Bottom : Δμ(z) = μ_model(z) − μ_ref(z)  [mag]  (absolute residual)
    where μ_ref is the best-fit curve of `reference_label` (defaults to
    the first model in the dict).

    The MAP absolute magnitude M is computed per model via _map_M() so that
    theory and data are on a consistent distance-modulus scale.

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
        Model label to use as μ_ref in the residual panel.
        Defaults to the first model label found.
    """
    import warnings

    z_plot = np.linspace(0.001, 1.8, 400)

    # ── locate Pantheonplus-style likelihood (needs _map_M) ───────────────────
    _pp_lik = None
    for _lrd in datasets.values():
        for _lbl, (_res, _pipe) in _lrd.items():
            for _lik in _pipe.likelihoods:
                if hasattr(_lik, '_map_M') and hasattr(_lik, 'zCMB'):
                    _pp_lik = _lik
                    break
            if _pp_lik is not None:
                break
        if _pp_lik is not None:
            break

    if _pp_lik is None:
        raise RuntimeError("[mu_plot] No marginalized-M (Pantheonplus) likelihood found "
                           "in any pipeline.")

    z_pp    = _pp_lik.zCMB
    zHEL    = _pp_lik.zHEL
    m_obs   = _pp_lik.m_obs
    mu_err_data = np.sqrt(np.diag(_pp_lik.cov))

    # req_plot: "mu" for theory curve on z_plot, "dL" at data z for MAP M
    req_plot = {"mu": {"z": z_plot}, "dL": {"z": z_pp}}

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

        # ── compute best-fit μ curves (pure distance modulus) ─────────────────
        bf_curves = {}   # label -> μ(z_plot)  [no M_B]
        M_B_map   = {}   # label -> MAP M_B scalar
        band_lo   = {}
        band_hi   = {}

        for label, (results, pipeline) in labeled_runs_dict.items():
            theta_bf = results.best_fit
            try:
                theory = pipeline.model.compute_theory(theta_bf, req_plot)
            except Exception as e:
                warnings.warn(f"[mu_plot] compute_theory failed for '{label}': {e}")
                continue
            if theory.invalid:
                warnings.warn(f"[mu_plot] best-fit theory invalid for '{label}'. Skipping.")
                continue

            # MAP M_B for this model
            dL_data = theory.eval("dL", z_pp)
            factor  = (1.0 + zHEL) / (1.0 + z_pp)
            mu_th_noM = 5.0 * np.log10(factor * dL_data) + 25.0
            M_B_map[label] = _pp_lik._map_M(mu_th_noM)

            # pure distance modulus (no M_B shift)
            bf_curves[label] = theory.eval("mu", z_plot)

            # credible band
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
                        samp.append(th.eval("mu", z_plot))
                except Exception:
                    pass

            if len(samp) >= 10:
                arr = np.array(samp)
                band_lo[label] = np.percentile(arr, 16, axis=0)
                band_hi[label] = np.percentile(arr, 84, axis=0)

        # Reference curve and its MAP M_B (used to convert data to distance moduli)
        _ref_local = _ref_label if _ref_label in bf_curves else next(iter(bf_curves))
        mu_ref    = bf_curves[_ref_local]
        M_B_ref   = M_B_map.get(_ref_local, next(iter(M_B_map.values())))

        # Data on distance-modulus scale: m_obs - M_B_ref
        mu_data_plot = m_obs - M_B_ref

        # ── top panel: distance modulus μ(z) ─────────────────────────────────
        ax_top.errorbar(z_pp, mu_data_plot, yerr=mu_err_data,
                        fmt="o", color="k", mfc="#D6860F", mec="k",
                        ms=3, ecolor="gray", elinewidth=0.6,
                        capsize=0, zorder=10, alpha=0.5, label=r"Pantheon$^+$")

        for label, mu_bf in bf_curves.items():
            col = col_map[label]
            ls  = ls_map[label]
            lw  = lw_map[label]
            ax_top.plot(z_plot, mu_bf, color=col, ls=ls, lw=lw,
                        label=label, zorder=5)
            if label in band_lo:
                ax_top.fill_between(z_plot, band_lo[label], band_hi[label],
                                    color=col, alpha=0.12, zorder=2)

        ax_top.set_ylabel(r"$\mu(z)$ [mag]", fontsize=13)
        ax_top.tick_params(labelsize=12)
        ax_top.set_xlim(z_plot[0], z_plot[-1])
        ax_top.text(0.03, 0.95, panel_title,
                    transform=ax_top.transAxes, ha="left", va="top",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        show_legend = (legend_panel == "all") or (legend_panel == panel_idx)
        if show_legend:
            ax_top.legend(fontsize=11, frameon=False,
                          bbox_to_anchor=(0.5, 1.25), loc="upper center",
                          ncol=4, labelspacing=0.25)

        # ── bottom panel: absolute residual Δμ = μ_model − μ_ref  [mag] ──────
        ax_res.axhline(0, color=col_map.get(_ref_local, "k"),
                       lw=0.8, ls="-", zorder=4)
        ax_res.axhspan(-0.05, 0.05, color="gray", alpha=0.08, zorder=1)

        # Data residual: (m_obs − M_B_ref) − μ_ref(z_data)
        mu_ref_at_data = np.interp(z_pp, z_plot, mu_ref)
        data_resid     = mu_data_plot - mu_ref_at_data
        ax_res.errorbar(z_pp, data_resid, yerr=mu_err_data,
                        fmt="o", color="k", mfc="#D6860F", mec="k",
                        ms=3, ecolor="gray", elinewidth=0.6,
                        capsize=0, zorder=10, alpha=0.5)

        for label, mu_bf in bf_curves.items():
            if label == _ref_local:
                continue
            col = col_map[label]
            ls  = ls_map[label]
            lw  = lw_map[label]
            resid = mu_bf - mu_ref
            ax_res.plot(z_plot, resid, color=col, ls=ls, lw=lw, zorder=5)
            if label in band_lo and label in band_hi:
                ax_res.fill_between(z_plot,
                                    band_lo[label] - mu_ref,
                                    band_hi[label] - mu_ref,
                                    color=col, alpha=0.15, zorder=2)

        ax_res.set_ylabel(r"$\Delta\mu$ [mag]", fontsize=11)
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