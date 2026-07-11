"""run_cosmix — Main entry point for the COSMIX inference pipeline.

Usage
-----
    python run_cosmix.py input.yaml

The YAML file specifies the model, likelihoods, sampler, convergence
strategy, and output options.  See input.yaml for a fully-annotated
example and README.md for the full list of supported keys.
"""
import numpy as np
from datetime import datetime, timezone
import importlib
import pkgutil
import sys, os

from cosmix.core.Pipeline import Pipeline
from cosmix.core.Registry import cosmix_registry

from cosmix.postprocessing.ResultsContainer import MCMCResults
from cosmix.postprocessing.Diagnostics import MCMCDiagnostics
from cosmix.postprocessing.Visualization import MCMCVisualization
from cosmix.postprocessing.Archive_.RunManifest import RunManifest
from cosmix.postprocessing.Archive_.RunArchive import RunArchive
from cosmix.postprocessing.Archive_.Serializers import YAML_load
from cosmix.postprocessing.MultiChainResults import MultiChainResults

from cosmix import theory, likelihoods, samplers

from cosmix.drivers.MultiChainDriver import MultiChainDriver
from cosmix.drivers.SingleChainConvergence import SingleChainStrategy, NestedStrategy
from cosmix.drivers.MultiFixedConvergence import MultiFixedStrategy
from cosmix.drivers.MultiAutoConvergence import MultiAutoStrategy


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"

#---------- AUTOMATIC MODEL / LIKELIHOOD / SAMPLER SCAN ----------#
def scan_modules():
    for package in [theory, likelihoods, samplers]:
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            try:
                importlib.import_module(f"{package.__name__}.{module_name}")
            except Exception as e:
                print(f"[run_cosmix] Skipping broken module {module_name} : {e}.")

#---------- REGISTRIES ----------#
PLOT_REGISTRY = {
        "trace": lambda viz: viz.trace(),
        "corner": lambda viz: viz.corner(),
        "residual": lambda viz: viz.residual(),
        "H": lambda viz: viz.plot_H(posterior_bands=True),
        "mu": lambda viz: viz.plot_mu(posterior_bands=True),
        "fsigma8": lambda viz: viz.plot_fs8(posterior_bands=True)
    }

CONVERGENCE_REGISTRY = {
    "single": SingleChainStrategy,
    "multi_fixed": MultiFixedStrategy,
    "multi_auto": MultiAutoStrategy
}
#--------------------#

def default_initial(pm):
    theta0 = []
    for p in pm.free_params:
        if hasattr(p.prior, "mean"):
            theta0.append(p.prior.mean)
        else:
            theta0.append(0.5*(p.prior.low + p.prior.high))

    return np.array(theta0)


def generate_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def main(yaml_path):

    scan_modules()

    config = YAML_load(yaml_path)

    # run id
    run_config = config.get("run", {})
    run_id = run_config.get("run_id", generate_run_id())

    archive = RunArchive()
    # Fail fast: check the run_id before any computation.
    # The directory is only created on successful completion (see archival block below).
    if config.get("outputs", {}).get("archive", False):
        archive.dry_run_id_check(run_id)

    # model
    model_name = config["model"]["name"]
    if model_name not in cosmix_registry.models:
        raise KeyError(f"[run_cosmix] Model {model_name} is not registered.")
    model_cls = cosmix_registry.models[model_name]

    # likelihoods
    likelihood_classes = []
    likelihood_kwargs = {}

    for L in config["likelihoods"]:
        likelihood_name = L["name"]
        if likelihood_name not in cosmix_registry.likelihoods:
            raise KeyError(f"[run_cosmix] Likelihood {likelihood_name} is not registered.")
        cls = cosmix_registry.likelihoods[likelihood_name]
        likelihood_classes.append(cls)
        likelihood_kwargs[cls] = L.get("options", {})

    
    # pipeline
    # Optional run-level parameter overrides: fix any free parameter to a value.
    # YAML example:
    #   parameters:
    #     lambda0:
    #       fixed: 1.5
    param_overrides = {}
    for pname, spec in config.get("parameters", {}).items():
        if "fixed" in spec:
            param_overrides[pname] = {"fixed": float(spec["fixed"])}

    pipeline = Pipeline(
        model_class=model_cls,
        likelihood_classes=likelihood_classes,
        likelihood_kwargs=likelihood_kwargs,
        param_overrides=param_overrides,
    )

    # ── Dataset overlap check ─────────────────────────────────────────────────
    # Controlled by `overlap_check:` in input.yaml.  Detects measurements from
    # the same galaxy survey appearing in multiple likelihoods (e.g. VIPERS fσ₈
    # in RSD *and* VIPERS E_G in EgStatistic).  Set abort_on_overlap: true to
    # stop the run hard when overlaps are found; false (default) only warns.
    overlap_cfg = config.get("overlap_check", {})
    if overlap_cfg.get("enabled", False):
        from cosmix.core.OverlapChecker import OverlapChecker
        z_tol        = float(overlap_cfg.get("z_tol", 0.01))
        abort_on_ovl = bool(overlap_cfg.get("abort_on_overlap", False))
        report = OverlapChecker(z_tol=z_tol).check(pipeline.likelihoods)
        report.pretty_print()
        if report.has_overlaps and abort_on_ovl:
            raise RuntimeError(
                "Overlapping datasets detected (see report above).  "
                "Remove duplicate survey measurements from one likelihood, "
                "or set abort_on_overlap: false in input.yaml to proceed."
            )

    # sampler
    sampler_config = config["sampler"]
    sampler_name = sampler_config["name"]
    if sampler_name not in cosmix_registry.samplers:
        raise KeyError(f"[run_cosmix] Sampler {sampler_name} is not registered.")
    sampler_cls = cosmix_registry.samplers[sampler_name]

    init_config = sampler_config.get("init", {})
    run_config = sampler_config.get("run", {})

    # Nested samplers (dynesty, polychord) take (pm, pipeline) and are
    # self-terminating; they don't use the convergence block.
    NESTED_SAMPLERS = {cosmix_registry.samplers.get("dynesty", cosmix_registry.samplers.get("polychord"))}
    is_nested = sampler_cls in NESTED_SAMPLERS

    if is_nested:
        sampler = sampler_cls(
            pm=pipeline.pm,
            pipeline=pipeline,
            **{k: v for k, v in init_config.items() if k != "random_seed"}
        )
        strategy = NestedStrategy(sampler=sampler, pipeline=pipeline)
    else:
        convergence_config = config.get("convergence", {"mode": "single"})
        mode = convergence_config["mode"]

        if mode == "single":
            # Single chain: the sampler is used directly for sampling, so the
            # Pre-Flight Optimizer must run here in the main process.
            sampler = sampler_cls(
                pm=pipeline.pm,
                lnpost=pipeline.lnposterior,
                nwalkers=None,
                random_seed=init_config.get("random_seed"),
                norm_func=pipeline.norm_terms_total
            )
            strategy = SingleChainStrategy(
                sampler=sampler,
                pipeline=pipeline,
                run_kwargs=run_config
            )
        else:
            # Multi-chain: the driver creates per-chain samplers internally and
            # runs the Pre-Flight Optimizer once per chain in each subprocess.
            # Creating a full sampler here would run a *redundant* optimizer in
            # the main process (~15–30 s), stalling chain startup for no benefit.
            # Instead, build a lightweight metadata stub used only for archiving.
            class _SamplerRef:
                pass
            sampler = _SamplerRef()
            sampler.__class__ = sampler_cls        # correct __class__.__name__
            sampler.nwalkers = max(5 * pipeline.pm.ndim, 20)

            driver = MultiChainDriver(
                sampler_cls=sampler_cls,
                sampler_kwargs=dict(
                    pm=pipeline.pm,
                    lnpost=pipeline.lnposterior,
                    nwalkers=None,
                    random_seed=init_config.get("random_seed", 42),
                    norm_func=pipeline.norm_terms_total
                ),
                nchains=convergence_config["nchains"],
                ncores=convergence_config["ncores"]
            )

            if mode == "multi_fixed":
                strategy = MultiFixedStrategy(
                    driver=driver,
                    run_kwargs=run_config,
                    rhat_tol=convergence_config.get("rhat_tol", 0.01),
                    ess_min=convergence_config.get("ess_min", 1000),
                    tau_factor=convergence_config.get("tau_factor", 50)
                )
            elif mode == "multi_auto":
                strategy = MultiAutoStrategy(
                    driver=driver,
                    run_kwargs=run_config,
                    rhat_tol=convergence_config.get("rhat_tol", 0.01),
                    ess_min=convergence_config.get("ess_min", 1000),
                    tau_factor=convergence_config.get("tau_factor", 50),
                    check_every=convergence_config.get("check_every",
                                                       run_config.get("nsteps", 500)),
                    max_steps=convergence_config.get("max_steps", 50000)
                )
            else:
                raise ValueError(f"[run_cosmix] Unknown convergence mode: {mode}")

    results = strategy.run()
    convergence_summary = strategy.summary()

    print("="*80)
    if strategy.is_converged():
        print("[run_cosmix] Chains appear to be converged.")
    else:
        print("[run_cosmix] Chains not converged.")
    if "detail" in convergence_summary:
        d = convergence_summary["detail"]
        print(f"  R-hat  : {'PASS' if d.get('rhat_ok') else 'FAIL'}  (max R-hat = {d.get('rhat_max', '?'):.4f})")
        if "ess_ok" in d:
            print(f"  ESS    : {'PASS' if d['ess_ok'] else 'FAIL'}  (min ESS = {d['ess_min_val']:.0f})")
        if "tau_ok" in d:
            print(f"  tau    : {'PASS' if d['tau_ok'] else 'FAIL'}  (max tau = {d['tau_max']:.1f})")
    if convergence_summary.get("mode") == "nested":
        logZ = convergence_summary.get("logZ")
        logZ_err = convergence_summary.get("logZ_err")
        if logZ is not None:
            print(f"  logZ   : {logZ:.3f} +/- {logZ_err:.3f}")
    print("="*80)
    #summary(results)

    if isinstance(results, MultiChainResults):
        mcmc_res = MCMCResults.from_multichain(results, pipeline, sampler_name)
    else:
        mcmc_res = results      # MCMCResults already (single, nested, or NestedStrategy)

    mcmc_dgn = MCMCDiagnostics(mcmc_res)
    mcmc_viz = MCMCVisualization(pipeline, mcmc_res)

    IC = mcmc_res.information_criteria(pipeline)
    mcmc_res.summary()

    # plots
    requested_plots = config["outputs"]["plots"]
    likelihod_plots = {}
    for L in pipeline.likelihoods:
        if L.get_plots() is not None:
            likelihod_plots.update(L.get_plots())

    all_plots = requested_plots | likelihod_plots
    figures = {}

    for name, enabled in all_plots.items():
        if not enabled:
            continue
        if name not in PLOT_REGISTRY:
            raise ValueError(f"[run_cosmix] Unknown plot type: {name}")

        fig = PLOT_REGISTRY[name](mcmc_viz)
        figures[name] = fig

    # archival
    if config["outputs"].get("archive", False):
        # Create the directory only now — run completed successfully.
        archive.create_run_dir(run_id)
        manifest = RunManifest.form_pipeline(
            pipeline=pipeline,
            sampler=sampler,
            convergence=convergence_summary,
            results=mcmc_res,
            run_id=run_id,
            config=config
        )
        archive.save_manifest(manifest)
        archive.save_chains(mcmc_res)
        archive.save_diagnostics(mcmc_res.diagnostics_dict(pipeline=pipeline))

        for name, fig in figures.items():
            archive.save_figure(fig, name)

def cli_entry():
    import sys
    if len(sys.argv) != 2:
        print("[run_cosmix] Usage: cosmix <input.yaml>")
        sys.exit(1)

    main(sys.argv[1])



if __name__ == "__main__":
    cli_entry()