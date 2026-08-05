"""
Pull plot utility
"""
import matplotlib.pyplot as plt
import numpy as np

# ── Band colours for grouped discrete panels (DES/DESI style) ────────────────
_BAND_COLORS = ["#eef2ff", "#ffffff"]   # alternating pale blue / white

def _group_by_second_line(labels):
    """Parse labels that may contain '\\n'.

    Returns
    -------
    types   : list[str]   — first line of each label (observable type)
    z_parts : list[str|None] — second line (z info), or None if absent
    groups  : list[(list[int], str|None)]
              consecutive runs sharing the same z_part, together with that
              z_part string (or None for simple labels)
    """
    parsed  = [lbl.split("\n", 1) for lbl in labels]
    types   = [p[0] for p in parsed]
    z_parts = [p[1] if len(p) == 2 else None for p in parsed]

    groups, cur_z, cur_indices = [], z_parts[0], [0]
    for idx in range(1, len(labels)):
        if z_parts[idx] == cur_z:
            cur_indices.append(idx)
        else:
            groups.append((cur_indices, cur_z))
            cur_z, cur_indices = z_parts[idx], [idx]
    groups.append((cur_indices, cur_z))

    return types, z_parts, groups


def pull_plot(
        labeled_runs, 
        figsize=(8,6), 
        save_fig=None, 
        dpi=500, 
        panel_labels=None, 
        title=None, 
        titlesize=None, 
        titleweight=None,
        leg_cols=3):
    """
    Per-dataset pull plot for multiple models.

    Discrete-observable panels (those whose likelihoods set ``_param_labels``)
    are rendered in the DES / DESI style:
      • alternating pale-blue / white bands per redshift group
      • observable type as the x-tick label  (rotation 45 °)
      • redshift group header below the tick labels

    Continuous panels (Pantheon+, RSD, BAO vs z) keep the standard
    "Redshift z" x-axis.

    Parameters
    ----------
    labeled_runs : dict  {label: (MCMCResults, Pipeline)}
    save_fig     : str or None   — file path for PDF/PNG output
    figwidth     : float         — total figure width in inches
    panel_labels : dict or None  — {lik.name: custom title}
    """
    if title is None:
        title = "Pull Plot"
    if titlesize is None:
        titlesize = 16
    if titleweight is None:
        titleweight = "bold"
    # ── 1. collect ordered observable UIDs ───────────────────────────────────
    obs_order, _seen = [], set()
    for label, (results, pipeline) in labeled_runs.items():
        theta  = results.best_fit
        theory = pipeline.model.compute_theory(theta, pipeline.requirements)
        for lik in pipeline.likelihoods:
            comp = lik.get_theory_components(theta, theory)
            if not comp:
                continue
            for obs_key in comp:
                uid = (lik.name, obs_key)
                if uid not in _seen:
                    _seen.add(uid)
                    obs_order.append(uid)

    if not obs_order:
        raise RuntimeError("[pull_plot] No likelihoods returned theory components.")

    # ── 2. pre-detect discrete-label panels ──────────────────────────────────
    discrete_labels = {}   # uid -> list[str]
    for (lik_name, obs_key) in obs_order:
        for _lbl, (res_, pipe_) in labeled_runs.items():
            for lik in pipe_.likelihoods:
                if lik.name == lik_name:
                    t_ = res_.best_fit
                    th_ = pipe_.model.compute_theory(t_, pipe_.requirements)
                    lik.get_theory_components(t_, th_)
                    if hasattr(lik, "_param_labels"):
                        discrete_labels[(lik_name, obs_key)] = lik._param_labels
                    break
            if (lik_name, obs_key) in discrete_labels:
                break

    # ── 3. layout ─────────────────────────────────────────────────────────────
    n_panels     = len(obs_order)
    model_colors = plt.cm.tab10(np.linspace(0, 0.7, len(labeled_runs)))
    markers      = ["o", "s", "^", "D", "v", "P", "*"]
    _panel_labels = panel_labels or {}

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=figsize,
        gridspec_kw={"wspace": 0.15},
    )
    if n_panels == 1:
        axes = [axes]
    ax_map = {uid: ax for uid, ax in zip(obs_order, axes)}

    # ── 4. scatter + error bars ───────────────────────────────────────────────
    for m_idx, (label, (results, pipeline)) in enumerate(labeled_runs.items()):
        theta  = results.best_fit
        theory = pipeline.model.compute_theory(theta, pipeline.requirements)
        col    = model_colors[m_idx]
        mrkr   = markers[m_idx % len(markers)]

        for lik in pipeline.likelihoods:
            comp = lik.get_theory_components(theta, theory)
            if not comp:
                continue
            for obs_key, (x, d_vec, th_vec, sigma) in comp.items():
                uid = (lik.name, obs_key)
                if uid not in ax_map or d_vec is None or sigma is None:
                    continue
                ax   = ax_map[uid]
                pull = (d_vec - th_vec) / sigma
                jitter = (m_idx - (len(labeled_runs) - 1) / 2) * 0.015 * (
                    x.max() - x.min() + 1e-6
                )
                ax.scatter(x + jitter, pull, color=col, marker=mrkr,
                           s=22, zorder=3,
                           label=label if obs_order.index(uid) == 0 else "")
                ax.errorbar(x + jitter, pull, yerr=np.ones_like(pull),
                            fmt="none", ecolor=col, elinewidth=0.7,
                            capsize=2, alpha=0.6)

    # ── 5. axes decoration ────────────────────────────────────────────────────
    for i, ((lik_name, obs_key), ax) in enumerate(ax_map.items()):
        uid = (lik_name, obs_key)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-1, 1, alpha=0.07, color="k")
        ax.axhspan(-2, 2, alpha=0.04, color="k")
        ax.set_title(_panel_labels.get(lik_name, lik_name), fontsize=14)
        ax.tick_params(axis="y", labelsize=13)

        #ax.set_ylabel(r"$(d - t)\,/\,\sigma$" if i == 0 else "", fontsize=10)

        if uid not in discrete_labels:
            # Standard continuous panel
            ax.set_xlabel(r"Redshift $z$", fontsize=14)
            ax.tick_params(axis="x", labelsize=13)
            continue

        # ── DES / DESI-style discrete panel ──────────────────────────────────
        pl = discrete_labels[uid]
        types, z_parts, groups = _group_by_second_line(pl)
        n  = len(pl)
        has_z_groups = any(z is not None for z in z_parts)

        ax.set_xticks(np.arange(n))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xlabel("")

        # Observable-type tick labels (rotation 45°, right-aligned)
        ax.set_xticklabels(types, rotation=45, ha="right", fontsize=12)
        ax.tick_params(axis="x", labelbottom=True, pad=2)

        if has_z_groups:
            # Alternating background bands + vertical separators
            trans = ax.get_xaxis_transform()   # x=data, y=axes-fraction
            for gi, (idxs, z_str) in enumerate(groups):
                lo, hi = min(idxs) - 0.45, max(idxs) + 0.45
                ax.axvspan(lo, hi, color=_BAND_COLORS[gi % 2],
                           zorder=0, alpha=1.0)
                if gi < len(groups) - 1:
                    ax.axvline(hi, color="#aaaaaa", lw=0.5, ls="--", zorder=1)
                # Redshift group header: centred just below tick labels
                if z_str is not None:
                    ctr = (min(idxs) + max(idxs)) / 2
                    ax.text(ctr, 0.13, z_str, transform=trans,
                            ha="center", va="top", fontsize=11, rotation=90,
                            color="#333333", fontstyle="italic")

    # ── 6. legend above figure ────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], marker=markers[i % len(markers)], color="w",
                   markerfacecolor=model_colors[i], markersize=7, label=lbl)
        for i, lbl in enumerate(labeled_runs.keys())
    ]
    fig.legend(handles=handles, fontsize=14, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=leg_cols)
    fig.suptitle(title, fontsize=titlesize, fontweight=titleweight, y=1.08)

    fig.tight_layout(rect=[0, 0.08, 1, 1.0])   # bottom margin for z headers

    if save_fig:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return fig