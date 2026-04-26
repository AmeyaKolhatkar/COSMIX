"""LikelihoodBase — abstract base class for all COSMIX likelihoods.

Every observational likelihood must inherit from LikelihoodBase and
implement at minimum:

    lnlike(theta, theory)   — returns the log likelihood for parameter
                              vector theta given a pre-computed TheoryCache.
    get_requirements()      — returns a dict mapping observable names to
                              the redshift arrays at which they are needed,
                              e.g. {"H": z_array, "dL": z_sn}.

Optionally override:
    declare_parameters(cls) — class method; returns a list of Parameter
                              objects for nuisance parameters owned by this
                              likelihood.
    norm_term()             — constant log-normalization contribution
                              (e.g. from analytic marginalization).
    get_theory_components() — provides (z, data, theory, sigma) tuples
                              for residual plots.

DOs
---
- Load and store data once in __init__.
- Build covariance / inv-covariance once in __init__.
- Only evaluate the likelihood in lnlike(); no I/O or MCMC logic there.

DON'Ts
------
- Never index theta directly (use pm.get_value(theta, name) instead).
- Never modify parameters or the ParameterManager.
"""

from abc import ABC, abstractmethod

#------------------------------
# Base Class Skeleton 
#------------------------------
class LikelihoodBase(ABC):
    name = "base"
    data_size = None
    produce_residuals = False

    def __init__(self, parameter_manager):
        self.pm = parameter_manager

    @abstractmethod
    def lnlike(self, theta, theory):
        pass
    
    @abstractmethod
    def get_requirements(self):
        pass

    def validate(self):
        pass 
    
    @classmethod
    def declare_parameters(cls):
        return []
    
    def supports_marginalization(self, name):
        """
        Return True if this likelihood supports marginalizing parameter `name`.
        Default: False.
        """
        return False
    
    def marginalization_terms(self, theory):
        """
        Return marginalization-related contributions.
        Default: None (no marginalization).
        """
        return None
    
    def get_theory_components(self, theta, theory, z_override=None):
        """
        Return a dict of components required for residual computation and plotting
            If z_override is None, use the likelihood's native data redshifts; otherwise compute the theory observables at z_override
        Returns
            x : array-like; independent variable, e.g. redshift
            d_vec : array-like; observed data-vector
            th_vec : array-like; predicted theory vector
            sigma : array-like or None; 1 sigma uncertainties (optional)

        Default : None

        {
            observable_name: (x, d_vec, th_vec, sigma)
        }
        """
        return None
    
    def norm_term(self):
        """
        Log-normalization for the likelihood.

        Must be overriden if the likelihood has data-dependent normalization or analytic marginalization
        """
        return 0.0
    
    """def get_observable_data(self, name, theta, theory):
        
        Returns (z, data, sigma) for observable name or None if not provided by the likelihood.
        
        comps = self.get_theory_components(theta, theory)
        return comps.get(name, None)"""
    
    def get_plots(self):
        """
        Returns dict:
            {plot_name: callable(viz) -> figure}
        """
        return {}
    

class GaussMargTerm:
    """ Generic container for likelihoods that use marginalization """
    def __init__(self, param_name, A, B, ln_norm=0.0):
        self.param_name = param_name
        self.A = float(A)
        self.B = float(B)
        self.ln_norm = float(ln_norm)    