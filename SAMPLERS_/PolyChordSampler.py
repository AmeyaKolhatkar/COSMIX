# PolyChord Sampler

import numpy as np
from pathlib import Path

try:
    import pypolychord
    from pypolychord.settings import PolyChordSettings
    _POLYCHORD_AVAILABLE = True
except ImportError:
    _POLYCHORD_AVAILABLE = False

from SAMPLERS_.NestedSamplingBase import NestedSamplerBase

class PolyChordSampler(NestedSamplerBase):
    """
    Wrapper for the PolyChord nested sampling algorithm.

    PolyChord uses slice sampling to explore the prior volume, which makes it
    significantly more efficient than MultiNest for high-dimensional or
    curved/degenerate posteriors. It also returns the Bayesian evidence logZ.

    Parameters
    ----------
    pm : ParameterManager
    pipeline : Pipeline
    nlive : int
        Number of live points (default 25*ndim, following PolyChord convention).
    num_repeats : int
        Slice sampling repeats per step (default 5*ndim). Increase for
        highly correlated posteriors.
    precision_criterion : float
        Evidence convergence criterion (default 0.001).
    base_dir : str
        Directory for PolyChord's temporary clustering files.
    file_root : str
        Prefix for PolyChord output files.
    """
    def __init__(
        self,
        pm,
        pipeline,
        nlive=None,
        num_repeats=None,
        precision_criterion=0.001,
        base_dir="chains",
        file_root="polychord_run"
    ):
        if not _POLYCHORD_AVAILABLE:
            raise ImportError(
                "[PolyChordSampler] pypolychord is not installed. "
                "Install it with: pip install -e DATA_/PolyChordLite"
            )
        super().__init__(pm, pipeline)
        self.nlive = nlive if nlive is not None else 25 * self.ndim
        self.num_repeats = num_repeats if num_repeats is not None else 5 * self.ndim
        self.precision_criterion = precision_criterion
        self.base_dir = base_dir
        self.file_root = file_root

    def _loglike_polychord(self, theta):
        """
        PolyChord requires loglike(theta) -> (logL, phi) where phi is a list
        of derived parameters. We return an empty derived list.
        """
        logL = self.loglike(theta)
        return logL, []

    _MIN_EQUAL_WEIGHTS = 50   # fall back to resampling if fewer equal-weight rows

    def _cleanup_output_files(self):
        """
        Delete stale PolyChord output files for the current base_dir/file_root
        before starting a fresh run.  Prevents column-count mismatches when
        ndim changes between runs (e.g. after adding a new likelihood/parameter).
        """
        import glob, os
        root = str(Path(self.base_dir) / self.file_root)
        patterns = [
            f"{root}.txt",
            f"{root}_equal_weights.txt",
            f"{root}_dead.txt",
            f"{root}_phys_live.txt",
            f"{root}.resume",
            f"{root}.stats",
            f"{root}.paramnames",
        ]
        for pattern in patterns:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except OSError:
                    pass
        # also clear per-cluster files
        cluster_dir = Path(self.base_dir) / "clusters"
        if cluster_dir.is_dir():
            for f in cluster_dir.glob(f"{self.file_root}_*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def _read_polychord_equal_weights(self, output):
        """
        Read equal-weight posterior samples from PolyChord text output.
        Format per row: weight, -2*loglike, theta_0, ..., theta_{ndim-1}, [derived...]

        If the equal-weights file has fewer than _MIN_EQUAL_WEIGHTS rows, OR if
        its column count doesn't match the expected ndim (stale file), we fall
        back to reading the weighted .txt file and resampling it with multinomial
        resampling so that downstream code always receives a usable sample array.
        """
        eq_path  = Path(f"{output.root}_equal_weights.txt")
        wt_path  = Path(f"{output.root}.txt")

        def _load_and_parse(path):
            if not path.exists():
                raise FileNotFoundError(f"[PolyChordSampler] Missing PolyChord output file: {path}")
            data = np.loadtxt(path, ndmin=2)
            n_param_cols = data.shape[1] - 2
            if n_param_cols < self.ndim:
                raise ValueError(
                    f"[PolyChordSampler] Stale or incompatible output file {path}: "
                    f"has {n_param_cols} parameter columns but model has ndim={self.ndim}. "
                    f"Delete the '{self.base_dir}' directory and re-run."
                )
            return data

        try:
            eq_data = _load_and_parse(eq_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"[PolyChordSampler] Warning: {e}. Falling back to weighted posterior file.")
            eq_data = None

        if eq_data is not None and len(eq_data) >= self._MIN_EQUAL_WEIGHTS:
            # Happy path – use the already-equal-weighted file directly.
            logl    = -0.5 * eq_data[:, 1]
            samples = eq_data[:, 2:2 + self.ndim]
            return samples, logl

        # Fallback: multinomial resample the weighted posterior file.
        if eq_data is not None:
            print(
                f"[PolyChordSampler] Only {len(eq_data)} equal-weight rows found; "
                f"resampling from weighted posterior file."
            )
        wt_data = _load_and_parse(wt_path)
        weights = wt_data[:, 0]
        logl_wt = -0.5 * wt_data[:, 1]
        params  = wt_data[:, 2:2 + self.ndim]

        weights = np.maximum(weights, 0.0)
        total   = weights.sum()
        if total == 0 or len(weights) == 0:
            raise RuntimeError(
                "[PolyChordSampler] Weighted posterior file has no usable samples. "
                "The run may have converged before collecting any posterior points."
            )
        weights /= total
        n_resample = max(len(weights), self._MIN_EQUAL_WEIGHTS)
        idx = np.random.choice(len(weights), size=n_resample, replace=True, p=weights)
        return params[idx], logl_wt[idx]

    def run(self):
        print(f"[PolyChordSampler] Initializing PolyChord with nlive={self.nlive}, num_repeats={self.num_repeats} . . .")
        self._cleanup_output_files()  # remove stale files from previous runs

        settings = PolyChordSettings(self.ndim, 0)    # nDerived = 0
        settings.nlive = self.nlive
        settings.num_repeats = self.num_repeats
        settings.precision_criterion = self.precision_criterion
        settings.base_dir = self.base_dir
        settings.file_root = self.file_root
        settings.read_resume = False
        settings.do_clustering = True
        settings.feedback = 1                         # 0=silent, 1=progress, 2=verbose

        print("[PolyChordSampler] Commencing nested sampling run . . .")
        output = pypolychord.run_polychord(
            loglikelihood=self._loglike_polychord,
            nDims=self.ndim,
            nDerived=0,
            settings=settings,
            prior=self.wrapper_prior_transform
        )

        logZ = output.logZ
        logZ_err = output.logZerr

        print(f"[PolyChordSampler] Sampling complete. Bayesian Evidence (logZ): {logZ:.3f} +/- {logZ_err:.3f}")

        # Read PolyChord posterior samples directly from text outputs.
        # This avoids getdist paramnames dependency in output.posterior.
        samples_eq, logl_eq = self._read_polychord_equal_weights(output)

        best_idx = np.argmax(logl_eq)
        best_fit = samples_eq[best_idx]

        return {
            "chain": samples_eq,
            "log_prob": logl_eq,
            "best_fit": best_fit,
            "logZ": float(logZ),
            "logZ_err": float(logZ_err),
            "raw_results": output
        }
