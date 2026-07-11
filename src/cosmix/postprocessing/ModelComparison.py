"""ModelComparison — information criteria and model comparison utilities.

Responsibilities
----------------
1. Information criteria: AIC, AICc, BIC, DIC
2. Reduced chi2 / chi2 per degree of freedom
3. Delta-IC tables comparing models against a reference
"""
import numpy as np

def delta_IC(IC_dicts, ref):
    """
    IC_dict: {model_name: IC_dict}
    ref_model: reference model key
    """
    ref_ic = IC_dicts[ref]

    deltas = {}
    for name, ic in IC_dicts.items():
        deltas[name] = {
            "Delta AIC": ic["AIC"] - ref_ic["AIC"],
            "Delta BIC": ic["BIC"] - ref_ic["BIC"],
            "Delta DIC": ic["DIC"] - ref_ic["DIC"]
        }

    return deltas