"""TheoryCache — lightweight, read-only store for pre-computed theory predictions.

The model computes all required observables once per parameter point and
stores them here.  Each likelihood then reads from the cache rather than
recomputing — this is important for performance because multiple likelihoods
may share the same underlying observable (e.g. both CC and RSD need H(z)).

Usage
-----
# Inside a cosmological model:
cache = TheoryCache()
cache.add("H",  z_grid, H_values)
cache.add("dL", z_sn,   dL_values)

# Inside a likelihood:
H_vals  = theory.get("H")["values"]
dL_at_z = theory.eval("dL", z_target)   # interpolates if needed
"""

import numpy as np
from scipy.interpolate import interp1d

#------------------------------
# Theory Cache 
#------------------------------
class TheoryCache:
    def __init__(self):
        self._store = {}
        self.invalid = False

    def mark_invalid(self):
        self.invalid = True
        return self

    def add(self, name, z, values):
        self._store[name] = {
            "z": np.asarray(z),                             # redshift grid
            "values": np.asarray(values)                    # precomputed numbers
        }

    def get(self, name):
        """ Likelihoods read theory using this """
        if name not in self._store:
            raise KeyError(f"Theory requirement {name} not available.")
        return self._store[name]
    
    def eval(self, name, z_new, kind="linear"):
        entry = self._store[name]
        if np.array_equal(z_new, entry["z"]):
            return entry["values"]
        
        return np.interp(z_new, entry["z"], entry["values"])
