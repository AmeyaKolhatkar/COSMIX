"""Pipeline — the central orchestration layer of COSMIX.

The Pipeline class wires together a cosmological model, a set of
likelihoods, and the shared ParameterManager.  After construction
it exposes three evaluation methods used by every sampler:

    lnprior(theta)      — log prior probability
    lnlike(theta)       — log likelihood
    lnposterior(theta)  — log posterior (= lnprior + lnlike)

Typical usage
-------------
pipeline = Pipeline(
    model_class=LCDM,
    likelihood_classes=[CosmicChronometers, Pantheonplus],
    likelihood_kwargs={
        Pantheonplus: {"data_file": "/path/to/data.dat"}
    }
)
print(pipeline.pm.free_names)  # ['H0', 'Omegam0', 'scriptM']
"""
import numpy as np
from collections import defaultdict
from CORE_.ParameterManager_ import ParameterManager
from CORE_.RequirementResolver import RequirementResolver

class Pipeline:
    """
    Orchestration layer for COSMIX.
    Owns parameters, model, likelihoods, and theory requirements.
    """
    def __init__(self, model_class, likelihood_classes, likelihood_kwargs=None):
        """
        model_class: subclass of CosmologyModelBase
        likelihood_classes: list of LikelihoodBase subclasses
        likelihood_kwargs: dict {LikelihoodClass: kwargs dict}
        """
        self.model_class = model_class
        self.likelihood_classes = likelihood_classes
        self.likelihood_kwargs = likelihood_kwargs or {}

        self.pm = ParameterManager()
        self.model = None
        self.likelihoods = []
        self.requirements = {}

        self._build()

    # -----------------------------
    # Core build logic
    # -----------------------------
    def _build(self):
        # 1. Register model parameters
        for p in self.model_class.declare_parameters():             
            self.pm.add_parameter(p)

        # 2. Register nuisance parameters 
        for lkcls in self.likelihood_classes:
            for p in lkcls.declare_parameters():
                self.pm.add_parameter(p)

        # 3. Freeze parameters
        self.pm.freeze()

        # 4. Instantiate likelihoods
        for lkcls in self.likelihood_classes:
            kwargs = self.likelihood_kwargs.get(lkcls, {})
            L = lkcls(pm=self.pm, **kwargs)
            self.likelihoods.append(L)

        # 5. Validate marginalization
        self._validate_marginalization()

        # 6. Instantiate model
        self.model = self.model_class(self.pm)

        # 7. Aggregate theory requirements
        resolver = RequirementResolver(self.likelihoods)
        self.requirements = resolver.resolve()

        # 8. Optional validation
        for L in self.likelihoods:
            L.validate()

    # -----------------------------
    # Marginalization validation
    # -----------------------------
    def _validate_marginalization(self):
        """
        Ensure all the marginalized parameters are supported by at least one active likelihood
        """
        for pname in self.pm.marginalized_names:
            supported = False
            for L in self.likelihoods:
                if L.supports_marginalization(pname):
                    supported = True
                    break
            if not supported:
                raise RuntimeError(f"{pname} is marked marginalized but no likelihood supports marginalizing it.")

    # -----------------------------
    # Requirement aggregation
    # -----------------------------
    """def _merge_requirements(self):
        requirements = {}

        for L in self.likelihoods:
            req = L.get_requirements()
            for key, z in req.items():
                z = np.asarray(z)
                if key not in requirements:
                    requirements[key] = z
                else:
                    requirements[key] = np.unique(
                        np.concatenate([requirements[key], z])
                    ) """


    def _collect_marginalization_terms(self, theory):
        """
        Collect marginalization related terms from likelihoods
        """
        terms = []
        for L in self.likelihoods:
            contri = L.marginalization_terms(theory)
            if contri is not None:
                terms.append(contri)
        return terms
    
    def _combine_marg_terms(self, terms):
        grouped = defaultdict(list)
        for t in terms:
            grouped[t.param_name].append(t)
        return grouped
    
    def _apply_gauss_marg(self, grouped_terms):
        ln_marg = 0.0
        for pname, terms in grouped_terms.items():
            A = sum(t.A for t in terms)
            B = sum(t.B for t in terms)
            ln_norm = sum(t.ln_norm for t in terms)

            # Gaussian marginalization: integral of exp(-0.5*A*(M-B/A)^2) dM = sqrt(2pi/A)
            ln_marg += 0.5 * (B * B) / A
            ln_marg += 0.5 * np.log(2.0 * np.pi / A)
            ln_marg += ln_norm

        return ln_marg


    # -----------------------------
    # Public Evaluation API
    # -----------------------------
    def lnprior(self, theta):
        return self.pm.lnprior(theta)
    

    def lnlike(self, theta):
        theory = self.model.compute_theory(theta, self.requirements)
        lnL = sum( L.lnlike(theta, theory) for L in self.likelihoods )
        if not np.isfinite(lnL):
            return -np.inf

        terms = self._collect_marginalization_terms(theory)
        if terms:
            grouped = self._combine_marg_terms(terms)
            lnL += self._apply_gauss_marg(grouped)

        return lnL
    
    def lnposterior(self, theta):
        try:
            theory = self.model.compute_theory(theta, self.requirements)
        except Exception:
            return -np.inf
        
        if getattr(theory, "invalid", False):
            return -np.inf
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        lnL = sum( L.lnlike(theta, theory) for L in self.likelihoods )
        if not np.isfinite(lnL):
            return -np.inf

        terms = self._collect_marginalization_terms(theory)
        if terms:
            grouped = self._combine_marg_terms(terms)
            lnL += self._apply_gauss_marg(grouped)

        return lp + lnL
    
    def norm_terms_total(self):
        return sum( L.norm_term() for L in self.likelihoods )

