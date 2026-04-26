# Model Comparison
"""
Responsibilities
    1. All the Information Criterion - AIC/AICc/BIC/DIC
    2. reduced chi2 / chi2 per degree of freedom
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