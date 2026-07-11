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
from scipy.interpolate import interp1d, RegularGridInterpolator

# ══════════════════════════════════════════════════════════════════════════════
# TheoryCache
# ══════════════════════════════════════════════════════════════════════════════
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
    
    def eval(self, name, z_new, kind="linear"):
        entry = self._store[name]
        if np.array_equal(z_new, entry["z"]):
            return entry["values"]
        
        return np.interp(z_new, entry["z"], entry["values"])
    
    def add_grid(self, name, k, z, values_2d):
        """Stores a 2D grid like P(k, z). values_2d has shape (len(k), len(z))"""
        self._store[name] = {
            "type": "grid_2d",
            "k": np.asarray(k),
            "z": np.asarray(z),
            "values": np.asarray(values_2d)         # shape (nk, nz)
        }

    def eval_grid(self, name, k_new, z_new):
        """Bilinear interpolation on a stored 2D grid like P(k,z)"""
        entry = self._store[name]
        if entry.get("type") != "grid_2d":
            raise KeyError(f"[TheoryCache] {name} is not a 2D Grid.")
        
        interp = RegularGridInterpolator(
            (np.log(entry["k"]), entry["z"]),
            np.log(entry["values"]),
            method="linear",
            bounds_error=False,
            fill_value=None
        )
        pts = np.column_stack([np.log(k_new), z_new])

        return np.exp(interp(pts))
        
    def get(self, name):
        """ Likelihoods read theory using this """
        if name not in self._store:
            raise KeyError(f"[TheoryCache] Theory requirement {name} not available.")
        return self._store[name]
