"""
Loading MCMCResult Object Utility
"""
import numpy as np
import yaml
from cosmix.postprocessing.ResultsContainer import MCMCResults 

# ══════════════════════════════════════════════════════════════════════════════

def load_results(run_dir):
    chain = np.load(f"{run_dir}/chain.npy")
    log_prob = np.load(f"{run_dir}/log_prob.npy")
    weights = np.load(f"{run_dir}/weights.npy") 

    with open(f"{run_dir}/manifest.yaml") as f:
        manifest = yaml.safe_load(f)

    return MCMCResults(
        chain=chain,
        log_prob=log_prob,
        tau=None,
        param_names=manifest["labels"]["names"],
        latex_names=manifest["labels"]["latex"],
        sampler_name=manifest["sampler"]["name"],
        acceptance=None,
        metadata=manifest,
        weights=weights
    )