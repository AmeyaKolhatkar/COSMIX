"""
S8 computing Utility
"""
import numpy as np
import json 

def s80(labelled_dirs, precision=None):
    S80 = {}
    for label, directory in labelled_dirs.items():

        with open(f"{directory}/diagnostics.json", "r") as f:
            loaded_json = json.load(f)

        om0 = loaded_json["parameters"]["Omegam0"]["mean"]
        sigma80 = loaded_json["parameters"]["sigma80"]["mean"]

        if precision is not None:
            S80[label] = round(sigma80 * (om0/0.3)**0.5, precision)
        else:
            S80[label] = sigma80 * (om0/0.3)**0.5

    return S80

