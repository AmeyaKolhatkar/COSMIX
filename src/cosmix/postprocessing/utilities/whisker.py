"""
Whisker Plot Utility
"""
import numpy as np
import matplotlib.pyplot as plt 

def whisker_plot(datasets, params=("sigma80",),
                 param_labels=(r"$\sigma_{80}$",),
                 shade_colors=("#ddeeff", "#ffeedd"),
                 save_fig=None, dpi=300):
    r"""
    Marginalized parameter constraint whisker plot.

    Shows weighted posterior mean +/- 1-sigma for each parameter.
    Dataset groups are distinguished by a background shade; models by
    color + marker.  Y-tick labels show model names only.

    Parameters
    ----------
    datasets     : dict  {dataset_label: labeled_runs_dict}
    params       : str or tuple[str]
    param_labels : str or tuple[str]
    shade_colors : tuple[str]   one background color per dataset group
    save_fig     : str or None
    dpi          : int
    """
    from matplotlib.patches import Patch as _Patch
    from matplotlib.lines   import Line2D as _L2D

    # ── normalise to tuples so single-param calls work ────────────────────────
    if isinstance(params, str):
        params = (params,)
    if isinstance(param_labels, str):
        param_labels = (param_labels,)

    n_params     = len(params)
    ds_labels    = list(datasets.keys())
    n_ds         = len(ds_labels)
    model_labels = list(list(datasets.values())[0].keys())
    n_models     = len(model_labels)

    colors  = plt.cm.tab10(np.linspace(0, 0.7, n_models))
    markers = ["o", "s", "^", "D", "v", "P"]

    group_gap = n_models-2        # vertical gap between dataset groups
    model_gap = 0.6             # vertical spacing within a group

    # overall y extent (data coords, before inversion)
    y_last = (n_ds - 1) * group_gap + (n_models - 1) * model_gap
    y_lo, y_hi = -0.3, y_last + 0.3

    fig, axes = plt.subplots(1, n_params,
                             figsize=(4.5 * n_params, 0.5 * n_models * n_ds),
                             sharey=False)
    if n_params == 1:
        axes = [axes]

    y_ticks  = []
    y_labels = []

    for ax_i, (ax, param, plabel) in enumerate(zip(axes, params, param_labels)):

        # ── dataset shading ───────────────────────────────────────────────────
        for ds_i, ds_label in enumerate(ds_labels):
            y_base    = ds_i * group_gap
            y_span_lo = y_base - 0.5
            y_span_hi = y_base + (n_models - 1) * model_gap + 0.5
            sc = shade_colors[ds_i % len(shade_colors)]
            ax.axhspan(y_span_lo, y_span_hi, color=sc, alpha=0.3, zorder=0)

        # ── error bars ────────────────────────────────────────────────────────
        for ds_i, (ds_label, labeled_runs) in enumerate(datasets.items()):
            y_base = ds_i * group_gap

            for m_i, (m_label, (results, _)) in enumerate(labeled_runs.items()):
                if param not in results.param_names:
                    continue
                p_idx = results.param_names.index(param)
                mean  = float(results.mean[p_idx])
                std   = float(results.std[p_idx])
                y     = y_base + m_i * model_gap

                ax.errorbar(mean, y, xerr=std,
                            fmt=markers[m_i % len(markers)],
                            color=colors[m_i], mec=colors[m_i],
                            ms=6, elinewidth=1.5, capsize=4, zorder=5)

                if ax_i == 0:
                    y_ticks.append(y)
                    y_labels.append(m_label)

        ax.set_xlabel(plabel, fontsize=14)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", length=0)   # remove y tick marks on all panels
        ax.set_ylim(y_hi, y_lo)              # inverted: larger value first

        if ax_i == 0:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([])
        else:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([])

    # ── legend: two rows ──────────────────────────────────────────────────────
    shade_handles = [
        _Patch(facecolor=shade_colors[i % len(shade_colors)], label=ds_labels[i])
        for i in range(n_ds)
    ]
    model_handles = [
        _L2D([0], [0], marker=markers[i % len(markers)], color="w",
             markerfacecolor=colors[i], markersize=7, label=lbl)
        for i, lbl in enumerate(model_labels)
    ]

    leg1 = fig.legend(handles=shade_handles, fontsize=10, frameon=False,
                      loc="upper center", bbox_to_anchor=(0.45, 1.1), ncol=2, columnspacing=5)
    fig.add_artist(leg1)
    fig.legend(handles=model_handles, fontsize=10, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, labelspacing=0.3)

    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
    plt.show()
