"""
Information Criterion and Evidence (ICE) Table Utility
"""
import yaml

def load_diagnostics_dict(run_dir):
    with open(f"{run_dir}/manifest.yaml") as fh:
        m = yaml.safe_load(fh)
    diag = m.get("diagnostics", {})
    ic   = diag.get("information_criteria", {})

    diag_dict = {
        "logZ":        diag.get("logZ"),
        "logZ_err":    diag.get("logZ_err"),
        "AIC":         ic.get("AIC"),
        "BIC":         ic.get("BIC"),
        "DIC":         ic.get("DIC"),
        "k":           ic.get("k"),
        "N":           ic.get("N"),
        "chi2_min":    ic.get("chi2_min"),
        "reduced_chi2": ic.get("reduced chi2"),
    }

    return diag_dict


def ICETable(run_dirs, reference_label, title=None):
    """
    Information Criterion and Evidence (ICE) table.

    Reads AIC, BIC, DIC, logZ and logZ_err directly from each run's
    manifest.yaml (written by run_cosmix.py at the end of every run).
    Reports Delta values relative to the chosen reference model, along
    with Jeffreys-scale interpretation for Delta logZ.

    Parameters
    ----------
    run_dirs : dict  {label: run_directory_path}
    reference_label : str
        Key in run_dirs to use as the reference (Delta = 0 row).
    title : str or None
        Optional heading printed above the table.

    Returns
    -------
    dict  {label: {"logZ", "logZ_err", "AIC", "BIC", "DIC",
                   "k", "N", "chi2_min", "reduced_chi2",
                   "Delta_logZ", "Delta_AIC", "Delta_BIC", "Delta_DIC"}}
    """
    # ── Jeffreys scale for Delta logZ (= ln Bayes factor) ────────────────────
    def jeffreys(dlnZ):
        a = abs(dlnZ)
        if   a < 1.0:  return "inconclusive"
        elif a < 2.5:  return "weak"
        elif a < 5.0:  return "moderate"
        else:          return "strong"

    # ── load diagnostics from each manifest ──────────────────────────────────
    raw = {}
    for label, run_dir in run_dirs.items():
        with open(f"{run_dir}/manifest.yaml") as fh:
            m = yaml.safe_load(fh)
        diag = m.get("diagnostics", {})
        ic   = diag.get("information_criteria", {})
        raw[label] = load_diagnostics_dict(run_dir=run_dir)

    # ── compute deltas relative to reference ─────────────────────────────────
    ref = raw[reference_label]
    results = {}
    for label, d in raw.items():
        entry = dict(d)
        entry["Delta_logZ"] = (d["logZ"] - ref["logZ"]) if (d["logZ"] is not None and ref["logZ"] is not None) else None
        entry["Delta_AIC"]  = (d["AIC"]  - ref["AIC"])  if (d["AIC"]  is not None and ref["AIC"]  is not None) else None
        entry["Delta_BIC"]  = (d["BIC"]  - ref["BIC"])  if (d["BIC"]  is not None and ref["BIC"]  is not None) else None
        entry["Delta_DIC"]  = (d["DIC"]  - ref["DIC"])  if (d["DIC"]  is not None and ref["DIC"]  is not None) else None
        results[label] = entry

    # ── print table ───────────────────────────────────────────────────────────
    sep = "─" * 125
    if title:
        print(f"\n{'':^105}")
        print(f"  {title}")
    print(sep)
    print(f"  {'Model':<28}  {'k':>3}  {'χ²_min':>9}  {'χ²_red':>7}  "
          f"{'ΔAIC':>8}  {'ΔBIC':>8}  {'ΔDIC':>8}  "
          f"{'ΔlnZ':>8}  {'±':>6}  {'Evidence':<14}")
    print(sep)
    for label, d in results.items():
        is_ref  = (label == reference_label)
        ref_tag = " ←" if is_ref else "  "
        dlnZ    = d["Delta_logZ"]
        # Reference row: suppress delta columns and evidence label
        if is_ref:
            dlnZ_s = f"{'—':>8}"
            daic   = f"{'—':>8}"
            dbic   = f"{'—':>8}"
            ddic   = f"{'—':>8}"
            evid   = "reference"
        else:
            dlnZ_s = f"{dlnZ:+.2f}" if dlnZ is not None else "  n/a"
            daic   = f"{d['Delta_AIC']:+.1f}" if d["Delta_AIC"] is not None else "  n/a"
            dbic   = f"{d['Delta_BIC']:+.1f}" if d["Delta_BIC"] is not None else "  n/a"
            ddic   = f"{d['Delta_DIC']:+.1f}" if d["Delta_DIC"] is not None else "  n/a"
            evid   = jeffreys(dlnZ) if dlnZ is not None else "—"
        err_s   = f"{d['logZ_err']:.3f}" if d["logZ_err"] is not None else "   —"
        chi2r   = f"{d['reduced_chi2']:.3f}" if d["reduced_chi2"] is not None else "  —"
        chi2m   = f"{d['chi2_min']:.1f}"    if d["chi2_min"]    is not None else "  —"
        print(f"  {label:<28}{ref_tag}  {d['k']:>3}  {chi2m:>9}  {chi2r:>7}  "
              f"{daic:>8}  {dbic:>8}  {ddic:>8}  "
              f"{dlnZ_s:>8}  {err_s:>6}  {evid:<14}")
    print(sep)
    print(f"  Reference: {reference_label}   |   N_data = {ref['N']}   |   "
          f"Positive ΔlnZ / negative ΔIC = favoured over reference")
    print(sep)

    return results