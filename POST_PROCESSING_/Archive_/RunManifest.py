"""RunManifest — serializable record of everything needed to reproduce a run.

A RunManifest is created at the end of every run and saved to
RUNS_/<run_id>/manifest.yaml.  It captures:

    run_id        — unique identifier string
    model         — model name and Python module path
    likelihoods   — list of likelihood names, modules, and options used
    labels        — parameter names (plain and LaTeX)
    parameters    — full parameter snapshot (prior type, bounds, status)
    sampler       — sampler class name and key hyperparameters
    convergence   — convergence diagnostics (R̂, ESS, τ, logZ, …)
    environment   — Python/package versions for reproducibility
    diagnostics   — AIC, BIC, DIC, acceptance rate, etc.

Use RunArchive to write and read manifests from disk.
"""

class RunManifest:
    def __init__(self, 
                 run_id: dict, 
                 model: dict, 
                 likelihoods: list, 
                 labels: dict,
                 parameters: dict, 
                 sampler: dict, 
                 convergence: dict,
                 environment: dict=None, 
                 diagnostics: dict=None):
        
        self.run_id = run_id
        self.model = model
        self.likelihoods = likelihoods
        self.labels = labels
        self.parameters = parameters
        self.sampler = sampler
        self.convergence = convergence
        self.environment = environment or {}
        self.diagnostics = diagnostics or {}

    @classmethod
    def form_pipeline(cls, pipeline, sampler, convergence, results, run_id):
        pm = pipeline.pm

        model_info = {
            "name": pipeline.model.name,
            "module": pipeline.model.__class__.__module__
        }

        likelihood_info = []
        for L in pipeline.likelihoods:
            likelihood_info.append(
                {
                    "name": L.name,
                    "module": L.__class__.__module__,
                    "options": getattr(L, "options", None)
                }
            )

        label_info = {
             "names": list(pm.free_names),
             "latex": list(pm.free_latex)
        }

        parameter_info = pm.snapshot()

        sampler_info = {
            "name": sampler.__class__.__name__,
            "ndim": pm.ndim,
            "nwalkers": getattr(sampler, "nwalkers", None),
        }

        convergence = {
            "mode": results.metadata["mode"],
            "nchains": results.metadata.get("nchains", 1),
            "criterion": results.metadata.get("criterion"),
            "threshold": results.metadata.get("threshold"),
            "chunks": results.metadata.get("chunks"),
            "converged": results.metadata.get("converged")
        }

        diagnostics = {
            "acceptance": _serialize(results.acceptance),
            "tau": _serialize(results.tau),
            "ess": _serialize(results.ess),
            "rhat": results.metadata.get("rhat") if hasattr(results, "metadata") else None,
            "logZ": results.metadata.get("logZ") if hasattr(results, "metadata") else None,
            "logZ_err": results.metadata.get("logZ_err") if hasattr(results, "metadata") else None,
        }

        return cls(
            run_id=run_id,
            model=model_info,
            likelihoods=likelihood_info,
            labels=label_info,
            parameters=parameter_info,
            sampler=sampler_info,
            convergence=convergence,
            diagnostics=diagnostics
        )
    
    def to_dict(self):
        return {
            "run_id": self.run_id,
            "model": self.model,
            "likelihoods": self.likelihoods,
            "labels": self.labels,
            "parameters": self.parameters,
            "sampler": self.sampler,
            "convergence": self.convergence,
            "environment": self.environment,
            "diagnostics": self.diagnostics
        }
    
def _serialize(x):
        if x is None:
            return None
        if hasattr(x, "tolist"):
            return x.tolist()
        return float(x)