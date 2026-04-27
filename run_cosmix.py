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
from CORE_.Pipeline import Pipeline

from POST_PROCESSING_.ResultsContainer import MCMCResults
from POST_PROCESSING_.Diagnostics import MCMCDiagnostics
from POST_PROCESSING_.Visualization import MCMCVisualization
from POST_PROCESSING_.Archive_.RunManifest import RunManifest
from POST_PROCESSING_.Archive_.RunArchive import RunArchive
from POST_PROCESSING_.Archive_.Serializers import YAML_load
from POST_PROCESSING_.MultiChainResults import MultiChainResults

from THEORY_.LCDM_ import LCDM
from THEORY_.fQ_Hybrid import fQHybrid
from THEORY_.fQ_EHybrid import fQEHybrid
from THEORY_.fQ_LSR import fQLSR
from THEORY_.fQ_LSR_IDE import fQLSRIDE
from THEORY_.fQ_Hybrid_IDE import fQHybridIDE

from LIKELIHOODS_.CosmicChronometers import CosmicChronometers
from LIKELIHOODS_.Pantheonplus import Pantheonplus
from LIKELIHOODS_.PantheonplusSH0ES import PantheonplusSH0ES
from LIKELIHOODS_.GW import GWStandardSiren
from LIKELIHOODS_.DESIDR2BAO import DESIDRIIBAO
from LIKELIHOODS_.H0priors import SH0ESprior, TRGBprior, H0LiCOWPrior
from LIKELIHOODS_.RSD import RedshiftSpaceDistortion
from LIKELIHOODS_.EgStatistic import EgStatistic
from LIKELIHOODS_.CompressedCMB import CompressedCMB

from SAMPLERS_.EmceeSampler import emceeSampler
from SAMPLERS_.DynestySampler import DynestySampler
from SAMPLERS_.PolyChordSampler import PolyChordSampler
from SAMPLERS_.MetropolisHastings import MHSampler

from DRIVERS_.MultiChainDriver import MultiChainDriver
from DRIVERS_.SingleChainConvergence import SingleChainStrategy, NestedStrategy
from DRIVERS_.MultiFixedConvergence import MultiFixedStrategy
from DRIVERS_.MultiAutoConvergence import MultiAutoStrategy

import sys, os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"


#---------- REGISTRIES ----------#
MODEL_REGISTRY = {
    "LCDM": LCDM,
    "fQ_Hybrid": fQHybrid,
    "fQ_EHybrid": fQEHybrid,
    "fQ_LSR": fQLSR,
    "fQ_LSR_IDE": fQLSRIDE,
    "fQ_Hybrid_IDE": fQHybridIDE
}

LIKELIHOOD_REGISTRY = {
    "CC": CosmicChronometers,
    "PP": Pantheonplus,
    "PPS": PantheonplusSH0ES,
    "GW": GWStandardSiren,
    "DDTB": DESIDRIIBAO,
    "SH0ES": SH0ESprior,
    "TRGB": TRGBprior,
    "H0LiCOW": H0LiCOWPrior,
    "RSD": RedshiftSpaceDistortion,
    "Eg": EgStatistic,
    "CompCMB": CompressedCMB
}

SAMPLER_REGISTRY = {
    "emcee": emceeSampler,
    "dynesty": DynestySampler,
    "polychord": PolyChordSampler,
    "mh": MHSampler,
}

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
    model_cls = MODEL_REGISTRY[config["model"]["name"]]

    # likelihoods
    likelihood_classes = []
    likelihood_kwargs = {}

    for L in config["likelihoods"]:
        cls = LIKELIHOOD_REGISTRY[L["name"]]
        likelihood_classes.append(cls)
        likelihood_kwargs[cls] = L.get("options", {})

    
    # pipeline
    pipeline = Pipeline(
        model_class=model_cls,
        likelihood_classes=likelihood_classes,
        likelihood_kwargs=likelihood_kwargs
    )

    # sampler
    sampler_config = config["sampler"]
    sampler_name = sampler_config["name"]
    sampler_cls = SAMPLER_REGISTRY[sampler_config["name"]]

    init_config = sampler_config.get("init", {})
    run_config = sampler_config.get("run", {})

    # Nested samplers (dynesty, polychord) take (pm, pipeline) and are
    # self-terminating; they don't use the convergence block.
    NESTED_SAMPLERS = {DynestySampler, PolyChordSampler}
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
                random_seed=init_config.get("random_seed")
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
                    random_seed=init_config.get("random_seed", 42)
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

    if isinstance(results, MultiChainResults):
        mcmc_res = MCMCResults.from_multichain(results, pipeline, sampler_name)
    else:
        mcmc_res = results      # MCMCResults already (single, nested, or NestedStrategy)

    mcmc_dgn = MCMCDiagnostics(mcmc_res)
    mcmc_viz = MCMCVisualization(pipeline, mcmc_res)

    IC = mcmc_res.information_criteria(pipeline)

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



if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("[run_cosmix] Usage: run_cosmix <input.yaml>")
    main(sys.argv[1])