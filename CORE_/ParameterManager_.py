"""ParameterManager — the single source of truth for all model parameters.

Responsibilities
----------------
1. Store parameter definitions (name, prior, role, status).
2. Assign a unique, immutable integer index to each *free* parameter so
   that theta vectors are always consistently ordered.
3. Provide fast value lookup:  pm.get_value(theta, "H0")
4. Compute the total log prior:  pm.lnprior(theta)
5. Transform unit-cube samples to physical parameters for nested samplers:
   pm.prior_transform(cube)
6. Distinguish free / fixed / marginalized parameters.

Key classes
-----------
Parameter     — a single named parameter with prior and role.
UniformPrior  — flat prior between [low, high].
GaussianPrior — truncated Gaussian prior (supports ppf for nested sampling).
ParameterManager — aggregates Parameter objects and exposes the API above.
"""

import numpy as np
from scipy.stats import truncnorm, norm as _norm


# ══════════════════════════════════════════════════════════════════════════════
# Parameter
# ══════════════════════════════════════════════════════════════════════════════
class Parameter:                                                        # This is only the minimal version
    def __init__(self, name, latex, prior, role, status="free", value=None, proposed_scale=0.05):     #  __init__ is a constructor which runs as the object is created
        self.name = name                                                # self refers to the specified parameter
        self.latex = latex
        self.prior = prior
        self.role = role                        # "cosmo" / "nuisance"
        self.status = status                    # "free" / "fixed" / "marginalized"
        self.value = value                      # fallback when status = "fixed"
        self.proposed_scale = proposed_scale       


# ══════════════════════════════════════════════════════════════════════════════
# Prior
# ══════════════════════════════════════════════════════════════════════════════
class Prior:
    def lnprob(self, x):
        raise NotImplementedError           # Every prior must define lnprob otherwise python will complain using NotImplementedError
    """ here, lnprob is not the posterior, i.e. lnprior + lnlike, but instead, it is the log prior probability density """

    def in_support(self, x):
        raise NotImplementedError
    
    def to_dict(self):
        return None

# ══════════════════════════════════════════════════════════════════════════════
# UniformPrior
# ══════════════════════════════════════════════════════════════════════════════
class UniformPrior(Prior):
    def __init__(self, low, high):
        self.low = low
        self.high = high 

    def lnprob(self, x):
        if self.low <= x <= self.high:
            return 0.0
        return -np.inf 
    
    def in_support(self, x):
        return self.low <= x <= self.high
    
    def ppf(self, q):
        """
        Percent Point function - Transforms a quantile (between 0 and 1) to the corresponding 
        physical value in the distribution. 
        """
        if np.isinf(self.low) or np.isinf(self.high):
            raise ValueError("Nested sampling strictly requires finite bounds for all parameters.")
        
        return self.low + q * ( self.high - self.low )
    
    def to_dict(self):
        return {
            "type": "Uniform",
            "low": self.low,
            "high": self.high
        }

# ══════════════════════════════════════════════════════════════════════════════
# GaussianPrior
# ══════════════════════════════════════════════════════════════════════════════
class GaussianPrior(Prior):
    """Gaussian prior.  Bounds are optional (default: unbounded).

    Usage
    -----
    GaussianPrior(mean=0.0, sig=1.0)                        # unbounded
    GaussianPrior(mean=70.0, sig=5.0, low=50.0, high=90.0) # truncated
    """
    def __init__(self, mean, sig, low=-np.inf, high=np.inf):
        self.mean = mean
        self.sig  = sig
        self.low  = low
        self.high = high

    def lnprob(self, x):
        if np.isfinite(self.low)  and x < self.low:  return -np.inf
        if np.isfinite(self.high) and x > self.high: return -np.inf
        return -0.5 * ((x - self.mean) / self.sig) ** 2

    def in_support(self, x):
        lo_ok = (not np.isfinite(self.low))  or x >= self.low
        hi_ok = (not np.isfinite(self.high)) or x <= self.high
        return lo_ok and hi_ok

    def ppf(self, q):
        """Inverse-CDF.  Uses truncated normal when bounds are finite."""
        if np.isfinite(self.low) and np.isfinite(self.high):
            a = (self.low  - self.mean) / self.sig
            b = (self.high - self.mean) / self.sig
            return truncnorm.ppf(q, a, b, loc=self.mean, scale=self.sig)
        return _norm.ppf(q, loc=self.mean, scale=self.sig)

    def to_dict(self):
        d = {"type": "Gaussian", "mean": self.mean, "sigma": self.sig}
        if np.isfinite(self.low):  d["low"]  = self.low
        if np.isfinite(self.high): d["high"] = self.high
        return d

# ══════════════════════════════════════════════════════════════════════════════
# ParameterManager
# ══════════════════════════════════════════════════════════════════════════════
class ParameterManager:
    def __init__(self):
        self._parameters = []               # ordered list - hence [] - of Parameter objects including free, fixed and marginalized 
        self._free_indices = {}             # name --> index mapping in theta, hence {} which is dict like
        self._fixed_values = {}             # name --> value mapping
        self._marginalized = set()          # unordered collection of names of marginalized params
        self._frozen = False                # safety lock
        # the underscore before signals internal use

    def add_parameter(self, param):
        if self._frozen:
            raise RuntimeError("[ParameterManager] Cannot add parameters after freeze.")
        
        if any(p.name == param.name for p in self._parameters):
            raise ValueError(f"[ParameterManager] Duplicate parameter : {param.name}")
        """ ValueError: occurs when a functions receives a correct data type but inappropriate/ invalid value
          for the operation at hand""" 

        if param.status == "free":
            self._free_indices[param.name] = len(self._free_indices)
        elif param.status == "fixed":
            if param.value is None:
                raise ValueError(f"[ParameterManager] Fixed parameter {param.name} needs a value")
            self._fixed_values[param.name] = param.value
        elif param.status == "marginalized":
            self._marginalized.add(param.name)
        else:
            raise ValueError(f"[ParameterManager] Unknown parameter status; '{param.status}'")
        
        self._parameters.append(param)

    def freeze(self):
        self._frozen = True

    def get_value(self, theta, name):
        if name in self._free_indices:
            return theta[self._free_indices[name]]
        elif name in self._fixed_values:
            return self._fixed_values[name]
        elif name in self._marginalized:
            raise RuntimeError(f"[ParameterManager] Parameter {name} is marginalized and hence its value cannot be explicitly obtained")
        else:
            raise KeyError(f"[ParameterManager] Unknown parameter {name}")
            # KeyError arises when a dict is being accessed using a key that does not exist

    """def param_index(self, name):
        
        Returns the index corresponding to the given parameter name.
        
        index = self._free_indices[name]
        self.free_names
        return """
    
    def lnprior(self, theta):
        total = 0.0
        for p in self._parameters:
            if p.status == "free":
                idx = self._free_indices[p.name]
                total += p.prior.lnprob(theta[idx])
        return total

    def is_marginalized(self, name):
        return name in self._marginalized
    
    def get_param(self, name):
        for p in self._parameters:
            if p.name == name:
                return p
        raise KeyError(name)
    
    def snapshot(self):
        """
        Returns a fully serilizable snapshot of all parameters. Method for the ARCHIVAL code.
        """
        params = {}
        for p in self._parameters:
            params[p.name] = {
                "role": p.role,
                "status": p.status,
                "prior": p.prior.to_dict(),
                "value": p.value
            }
        
        return params
    
    def prior_transform(self, cube):
        """
        Transforms a unit hypercube [0, 1]^N into physical parameters.
        Some samplers modify the cube in place, others expect a return vector.
        We return a new numpy array for universal safety.
        """
        cube = np.asarray(cube, dtype=float)
        if cube.shape[0] != self.ndim:
            raise ValueError(
                f"[ParameterManager] prior_transform expected cube of size {self.ndim}, got {cube.shape[0]}"
            )

        theta = np.empty(self.ndim, dtype=float)
        for p in self._parameters:
            if p.status != "free":
                continue

            idx = self._free_indices[p.name]
            if not hasattr(p.prior, "ppf"):
                raise AttributeError(
                    f"[ParameterManager] Prior for '{p.name}' must define ppf(q) for nested sampling transforms"
                )
            theta[idx] = p.prior.ppf(cube[idx])
        
        return theta
            
    
    # META DATA
    @property
    def free_names(self):
        return [p.name for p in self._parameters if p.status == "free"]
    
    @property
    def free_latex(self):
        return [p.latex for p in self._parameters if p.status == "free"]
    
    @property
    def free_params(self):
        return [p for p in self._parameters if p.status == "free"]
    
    @property
    def fixed_names(self):
        return list(self._fixed_values.keys())
    
    @property
    def marginalized_names(self):
        return list(self._marginalized)
    
    @property
    def ndim(self):
        return len(self._free_indices)
    
