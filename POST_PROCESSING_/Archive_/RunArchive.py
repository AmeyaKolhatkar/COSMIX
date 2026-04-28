# RunArchive
"""
Dependancies allowed:
    - pathlib.Path
    - numpy
    - JSON
    - YAML (via serializers)

Dependancies absolutely not allowed:
    - Pipeline
    - Likelihood
    - Models
    - Samplers
    - emcee
    - GetDist
"""
# RUNS_ lives at the repository root, two levels above this file.
from pathlib import Path
from POST_PROCESSING_.Archive_.Serializers import YAML_dump, JSON_dump, array_dump
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
Default_base = _REPO_ROOT / "RUNS_"

class RunArchive:
    def __init__(self, base_dir=Default_base):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"This path does not exist.")
        
        self.run_dir = None
        self.fig_dir = None

    
    def create_run_dir(self, run_id):
        """
        Create RUNS_/<parent_dirs>/run_<leaf>/ and a figures/ subdirectory.

        run_id may contain forward slashes to create a nested layout:
            "run_1"          -> RUNS_/run_run_1/
            "EXP1/run_1"     -> RUNS_/EXP1/run_run_1/
            "EXP1/v2/run_1" -> RUNS_/EXP1/v2/run_run_1/

        The 'run_' prefix is applied to the leaf segment only, and only
        when it does not already start with 'run'.

        Raises FileExistsError if the directory already exists.
        """
        parts = Path(run_id).parts          # splits on both / and \
        *parents, leaf = parts
        leaf_dir = leaf if leaf.startswith("run") else f"run_{leaf}"
        self.run_dir = self.base_dir.joinpath(*parents, leaf_dir) if parents else self.base_dir / leaf_dir

        if self.run_dir.exists():
            raise FileExistsError(f"[RunArchive] The directory {self.run_dir} already exists.")

        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.fig_dir = self.run_dir / "figures"
        self.fig_dir.mkdir()

    def dry_run_id_check(self, run_id):
        """
        Check whether the run directory for run_id already exists.
        Raises FileExistsError immediately so the user learns of the
        conflict before any expensive computation starts.
        Supports the same nested path syntax as create_run_dir.
        """
        parts = Path(run_id).parts
        *parents, leaf = parts
        leaf_dir = leaf if leaf.startswith("run") else f"run_{leaf}"
        path = self.base_dir.joinpath(*parents, leaf_dir) if parents else self.base_dir / leaf_dir

        if path.exists():
            raise FileExistsError(
                f"[RunArchive] Run directory {path} already exists. "
                f"Choose a different run_id or remove the existing directory."
            )


    def _check_initialized(self):
        if self.run_dir is None:
            raise RuntimeError("Run directory not initialized. Call create_run_directory() first.")
        

    def save_manifest(self, manifest):
        """
        Serialize RunManifest
        Write manifest.yaml without any modification
        """
        self._check_initialized()
        manifest_dict = manifest.to_dict()
        YAML_dump(manifest_dict, self.run_dir / "manifest.yaml")


    def save_arrays(self, chain, log_prob, weights=None):
        """
        Saves chain.npy and log_prob.npy.  For nested-sampling runs, also saves
        weights.npy when importance weights are provided — these are required by
        the dataset-consistency / tension-probability analysis in DatasetTension.ipynb.
        """
        self._check_initialized()
        array_dump(chain, self.run_dir / "chain.npy")
        array_dump(log_prob, self.run_dir / "log_prob.npy")
        if weights is not None:
            array_dump(weights, self.run_dir / "weights.npy")


    def save_arrays_multi(self, mcres, indent=2):
        # mcres.chain is a list of arrays, one per chain: (nsamples, ndim)
        self._check_initialized()
        info = {"nchains": mcres.nchains, "param_names": mcres.param_names}
        for i, arr in enumerate(mcres.chains):
            array_dump(arr, self.run_dir / f"chain_{i}.npy")
        for i, lp in enumerate(mcres.log_probs):
            array_dump(lp, self.run_dir / f"log_prob_{i}.npy")
        
        JSON_dump(info, self.run_dir / "chains_meta.json", indent=indent)     

    def save_chains(self, results):
        if hasattr(results, "chains"):
            self.save_arrays_multi(results)
        else:
            self.save_arrays(results.chain, results.log_prob,
                             weights=getattr(results, "weights", None))


    def save_diagnostics(self, diagnostics):
        """
        Dump JSON without any computation; this is just recording numbers
        """
        self._check_initialized()
        JSON_dump(diagnostics, self.run_dir / "diagnostics.json")


    def save_figure(self, fig, name, ext="pdf"):
        """
        Save into /figures, enforce consistent naming and close figure after saving.
        """
        self._check_initialized()
        path = self.fig_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=1000)
        fig.clear()