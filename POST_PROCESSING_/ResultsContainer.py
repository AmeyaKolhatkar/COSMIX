"""ResultsContainer — stores and analyses posterior samples from any sampler.

MCMCResults holds the equal-weighted sample chain, log-probabilities,
best-fit parameters, and metadata from a completed run.  It provides:

    information_criteria(pipeline)
        Computes AIC, BIC, and DIC.  Automatically handles the difference
        between MCMC output (log_prob = lnlike + lnprior) and nested
        sampling output (log_prob = lnlike only).

    summary()
        Prints a parameter table with mean ± std and best-fit values.

Factory class methods (called by convergence strategies):
    MCMCResults.from_sampler_output(...)   — for emcee / MH output
    MCMCResults.from_nested_output(...)    — for Dynesty / PolyChord output
"""
import numpy as np

class MCMCResults:
    """
    Container for MCMC output
    """
    def __init__(self, chain, log_prob, tau, param_names, latex_names, sampler_name, acceptance=None, metadata=None):
        self.chain = np.asarray(chain)
        self.log_prob = np.asarray(log_prob)
        self.tau = None if tau is None else np.asarray(tau)
        self.param_names = list(param_names)
        self.latex_names = list(latex_names)
        self.sampler_name = sampler_name
        self.acceptance = acceptance
        self.metadata = metadata or {}

        self._validate()
        self._compute_basic_stats()

    def _validate(self):
        if self.chain.ndim != 2:
            raise ValueError("Chain must be 2D: (nsamples, ndim)")
        if self.chain.shape[0] != self.log_prob.shape[0]:
            raise ValueError("Chain and log prob length mismatch")
        if self.chain.shape[1] != len(self.param_names):
            raise ValueError("Parameter name mismatch")
        
    def _compute_basic_stats(self):
        idx = np.argmax(self.log_prob)
        self.best_fit = self.chain[idx]
        self.mean = np.mean(self.chain, axis=0)
        self.std = np.std(self.chain, axis=0)

    def information_criteria(self, pipeline, store=True):
        # free parameters
        k = len(self.param_names)
        # N_total
        N = sum(L.data_size for L in pipeline.likelihoods)
        # log-likelihood samples:
        # For MCMC:          log_prob = lnlike + lnprior  =>  lnlike = log_prob - lnprior - norm
        # For nested (NS):   log_prob = lnlike            =>  lnlike = log_prob - norm
        norm_total = pipeline.norm_terms_total()
        if self.metadata.get("mode") == "nested":
            lnL_samples = self.log_prob - norm_total
        else:
            lnprior_samples = np.array(
                [pipeline.lnprior(theta) for theta in self.chain]
            )
            lnL_samples = self.log_prob - lnprior_samples - norm_total
        
        # lnL_max
        lnL_max = np.max(lnL_samples)

        # dof
        dof = N - k
        # reduced chi2
        chi2_red = -2.0 * lnL_max / dof
        # AIC
        AIC = -2.0 * lnL_max + 2.0 * k
        # BIC
        BIC = -2.0 * lnL_max + k * np.log(N)
        # DIC
        D_samples = -2.0 * lnL_samples
        D_bar = np.mean(D_samples)
        #theta_mean = self.mean
        lnL_mean = pipeline.lnlike(self.best_fit) - norm_total
        D_mean = -2.0 * lnL_mean
        DIC = 2.0 * D_bar - D_mean

        IC = {
            "k": k,
            "N": N,
            "dof": dof,
            "reduced chi2": float(chi2_red),
            "AIC": float(AIC),
            "BIC": float(BIC),
            "DIC": float(DIC),      
        }

        if store:
            self._information_criteria = IC

        return IC


    def summary(self):
        print(f"Sampler: {self.sampler_name}")
        if self.acceptance is not None:
            print(f"Acceptance fraction: {self.acceptance:.3f}")
        print("-"*75)
        for i, name in enumerate(self.param_names):
            print(
                f"{name:10s}"
                f"{self.mean[i]:.4f} +- {self.std[i]:.4f}"
                f"      Best: {self.best_fit[i]:.4f}"
            )

    @property
    def lnL_max(self):
        return np.max(self.log_prob)
    
    @property
    def ess(self):
        """
        Effective Sample Size per parameter
        """
        if self.tau is None:
            return None
    
        return self.chain.shape[0] / self.tau
    
    @classmethod
    def from_nested_output(cls, results, pipeline, sampler_name):
        """
        Build an MCMCResults from the dict returned by DynestySampler or
        PolyChordSampler.  The equal-weighted posterior samples play the role
        of the MCMC chain; τ is not meaningful for nested sampling so we store
        None and instead carry logZ in metadata.
        """
        return cls(
            chain=results["chain"],
            log_prob=results["log_prob"],
            tau=None,
            param_names=pipeline.pm.free_names,
            latex_names=pipeline.pm.free_latex,
            sampler_name=sampler_name,
            acceptance=None,
            metadata={
                "model": pipeline.model.name,
                "likelihoods": [L.name for L in pipeline.likelihoods],
                "mode": "nested",
                "logZ": results.get("logZ"),
                "logZ_err": results.get("logZ_err"),
            }
        )

    @classmethod
    def from_sampler_output(cls, results, pipeline, sampler_name):
        return cls(
            chain=results["chain"],
            log_prob=results["log_prob"],
            tau=results["tau"],
            param_names=pipeline.pm.free_names,
            latex_names=pipeline.pm.free_latex,
            sampler_name=sampler_name,
            acceptance=results["acceptance"],
            metadata={
                "model": pipeline.model.name,
                "likelihoods": [L.name for L in pipeline.likelihoods],
                "mode": "singlechain"
            }
        )
    
    @classmethod
    def from_multichain(cls, mcres, pipeline, sampler_name):
        chain = mcres.stacked_chain
        log_prob = mcres.stacked_log_prob

        best_idx = np.argmax(log_prob)
        theta_best = chain[best_idx]

        return cls(
            chain=chain,
            log_prob=log_prob,
            tau=mcres.tau,
            param_names=pipeline.pm.free_names,
            latex_names=pipeline.pm.free_latex,
            sampler_name=sampler_name,
            acceptance=None,    # per chain, not scalar
            metadata={
                "model": pipeline.model.name,
                "likelihoods": [L.name for L in pipeline.likelihoods],
                "mode": "multichain",
                "nchains": mcres.nchains,
                "rhat": mcres.rhat.tolist(),
                "ess": mcres.ess.tolist()
            }
        )
    
    def diagnostics_dict(self, pipeline=None):
        d = {
            "tau": _serialize(self.tau),
            "ess": _serialize(self.ess),
            "acceptance": _serialize(self.acceptance),
            "rhat": self.metadata.get("rhat", None),
            "information_criteria": self._information_criteria
        }
        if self.metadata.get("mode") == "nested":
            logZ     = self.metadata.get("logZ")
            logZ_err = self.metadata.get("logZ_err")
            d["logZ_raw"]  = logZ       # raw dynesty value (includes norm constants)
            d["logZ_err"]  = logZ_err
            if pipeline is not None and logZ is not None:
                norm = pipeline.norm_terms_total()
                d["logZ_physical"]     = float(logZ) - float(norm)   # physical Bayesian evidence
                d["logZ_physical_err"] = logZ_err
                d["norm_terms_total"]  = float(norm)
        return d

def _serialize(x):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return float(x)