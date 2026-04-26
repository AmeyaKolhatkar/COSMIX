"""RequirementResolver — merges redshift grids from all active likelihoods.

Each likelihood declares which theory observables it needs and at which
redshifts.  This module aggregates those requests across all likelihoods,
deduplicates them, and produces a single canonical requirements dict:

    {
        "H":       {"z": merged_z_array},
        "dL":      {"z": merged_z_array},
        "fsigma8": {"z": merged_z_array},
        ...
    }

The Pipeline passes this dict to the cosmological model, which evaluates
all observables in one pass and stores the results in a TheoryCache.
"""
import numpy as np
from collections import defaultdict

class RequirementResolver:
    def __init__(self, likelihood):
        """
        likelihoods: list
            list of instantiated likelihoods
        """
        self.likelihoods = likelihood

    def resolve(self):
        """
        resolves and aggregates requirements from all likelihoods

        returns:
        requirements: dict

        """
        raw = self._collect_raw_requirements()
        merged = self._merge_raw_requirements(raw)

        return merged
    
    def _collect_raw_requirements(self):
        raw = []
        for L in self.likelihoods:
            req = L.get_requirements()
            if not isinstance(req, dict):
                raise TypeError(f"{L.name}.get_requirements() must return a dict")
            
            raw.append(req)

        return raw
    
    def _merge_raw_requirements(self, raw):
        merged = {}

        for req in raw:
            for qty, z in req.items():
                z = np.asarray(z, dtype=float)

                if qty not in merged:
                    merged[qty] = {
                        "z": z.copy(),
                        "metadata": {}
                    }
                else:
                    merged[qty]["z"] = self._merge_grids(merged[qty]["z"], z)

        for q in merged:
            merged[q]["z"] = np.unique(np.sort(merged[q]["z"]))

        return merged
    
    
    @staticmethod
    def _merge_grids(z1, z2):
        """
        merge two redshift grids safely
        """

        return np.concatenate([z1, z2])
    
def single_requirement(name, z):

    return {name: {"z": np.asarray(z), "meta": {}}}