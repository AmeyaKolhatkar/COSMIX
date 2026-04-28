"""MultiChainDriver — runs multiple independent chains in parallel.

Launches nchains independent sampler instances using Python multiprocessing.
Each chain gets a unique random seed for decorrelated sampling.  Walkers
are initialized near the MAP with small jitter.

Provides two run modes:

    run(run_kwargs)
        Single pass — runs all chains once and returns a MultiChainResults.
        Used by MultiFixedStrategy.

    continuous_auto_run(run_kwargs, ...)
        Cobaya-style continuous run — chains start once (single Pre-Flight
        Optimizer call per chain), then run indefinitely in check_every-step
        intervals.  After each interval all chains are pooled and R̂, ESS,
        and τ are checked.  Runs until convergence or max_steps is reached.
        Used by MultiAutoStrategy.
"""
from POST_PROCESSING_.MultiChainResults import MultiChainResults
from multiprocessing import Pool
import numpy as np
from functools import partial
import os
from tqdm import tqdm

class MultiChainDriver:
    def __init__(self, sampler_cls, sampler_kwargs, nchains=4, ncores=4):
        self.sampler_cls = sampler_cls
        self.sampler_kwargs = sampler_kwargs
        self.nchains = nchains
        self.ncores = ncores

    def _make_sampler(self, chain_id, initial_walkers=None):
        kwargs = dict(self.sampler_kwargs)
        kwargs["random_seed"] = 1000 + chain_id
        kwargs["verbose"] = (chain_id == 0)  # chain 0 prints Pre-Flight chi2; others stay silent

        if initial_walkers is not None:
            # Resume from provided positions — skip Pre-Flight Optimizer
            kwargs["initial_walkers"] = initial_walkers
            return self.sampler_cls(**kwargs)

        # Fresh initialization: run Pre-Flight Optimizer then apply multi-chain jitter
        sampler = self.sampler_cls(**kwargs)

        if hasattr(sampler, "p0") and sampler.p0 is not None:
            valid_p0 = []

            if sampler.pm.free_params and hasattr(sampler.pm.free_params[0], "proposed_scale"):
                scales = [p.proposed_scale for p in sampler.pm.free_params]
                scale_array = np.array(scales)
            else:
                scale_array = np.ones(sampler.ndim) * 1e-4

            for point in sampler.p0:
                attempts = 0
                max_attempts = 100
                cand = point

                while attempts < max_attempts:
                    cand_test = point + scale_array * np.random.randn(sampler.ndim)

                    if np.isfinite(sampler.pm.lnprior(cand_test)):
                        if np.isfinite(sampler.lnpost(cand_test)):
                            cand = cand_test
                            break

                    attempts += 1

                valid_p0.append(cand)

            sampler.p0 = np.array(valid_p0)

        return sampler

    # ------------------------------------------------------------------
    # Internal worker called in each subprocess
    # ------------------------------------------------------------------
    def _run_single(self, chain_id, run_kwargs):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_NUM_THREADS"] = "1"

        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        sampler = self._make_sampler(chain_id)
        return sampler.run(**run_kwargs)

    def _run_single_with_state(self, args):
        """Worker for continuous_auto_run.  args = (chain_id, run_kwargs, initial_walkers)."""
        chain_id, run_kwargs, initial_walkers = args

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_NUM_THREADS"] = "1"

        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        sampler = self._make_sampler(chain_id, initial_walkers=initial_walkers)
        result = sampler.run(**run_kwargs)
        return result["chain"], result["log_prob"], result["final_state"]

    # ------------------------------------------------------------------
    # Single-pass run (MultiFixedStrategy)
    # ------------------------------------------------------------------
    def run(self, run_kwargs):
        print(f"[MultiChainDriver] Starting {self.nchains} chains (Pre-Flight Optimizer + sampling running in parallel) . . .")
        worker = partial(self._run_single, run_kwargs=run_kwargs)
        if self.ncores > 1:
            with Pool(self.ncores) as pool:
                results = pool.map(worker, range(self.nchains))
        else:
            results = [worker(i) for i in range(self.nchains)]

        chains = [r["chain"] for r in results]
        log_prob = [r["log_prob"] for r in results]

        return MultiChainResults(
            chains=chains,
            log_probs=log_prob,
            param_names=self.sampler_kwargs["pm"].free_names
        )

    # ------------------------------------------------------------------
    # Cobaya-style continuous run (MultiAutoStrategy)
    # ------------------------------------------------------------------
    def continuous_auto_run(self, run_kwargs, rhat_tol=0.01, ess_min=1000,
                            tau_factor=50, check_every=500, max_steps=50000):
        """
        Run chains continuously, checking convergence every check_every steps.

        Unlike the old auto_run, chains are NEVER restarted.  The Pre-Flight
        Optimizer runs exactly once per chain (on the first interval).  Walker
        positions are passed back into the next interval so sampling is truly
        continuous.  All samples accumulate across intervals.

        Parameters
        ----------
        run_kwargs : dict
            Passed to sampler.run().  burn_in is applied on the first interval
            only; subsequent intervals always use burn_in=0.
        check_every : int
            Steps per interval.  Convergence is assessed after each interval.
        max_steps : int
            Hard stop on total accumulated steps per chain.
        """
        states = [None] * self.nchains      # None → Pre-Flight Optimizer on first call
        all_chains    = [[] for _ in range(self.nchains)]
        all_log_probs = [[] for _ in range(self.nchains)]
        total_steps = 0
        iteration   = 0
        merged      = None

        print(f"[MultiChainDriver] Starting {self.nchains} chains "
              f"(continuous mode, checking every {check_every} steps) . . .")

        # Dynesty-style bar: no fill, just a live counter that rewrites one line
        bar = tqdm(
            total=None,
            unit="it",
            bar_format="{n_fmt}it [{elapsed}, {rate_fmt}{postfix}]",
            dynamic_ncols=True,
        )
        bar.set_postfix_str(
            f"chains: {self.nchains} | max R\u0302: --- | min ESS: --- | status: initializing",
            refresh=False,
        )

        with Pool(self.ncores) if self.ncores > 1 else _NullPool() as pool:
            while total_steps < max_steps:
                # Build per-interval kwargs
                chunk_kw = dict(run_kwargs)
                chunk_kw["nsteps"] = check_every
                # Always suppress burn_in in continuous mode — the Pre-Flight
                # Optimizer already places walkers near the MAP, so no burn-in
                # is needed.  Inheriting burn_in from run_kwargs causes the first
                # interval to run burn_in + check_every steps instead of just
                # check_every, stalling the run invisibly.
                chunk_kw["burn_in"] = 0
                # Suppress per-chain progress bars — tqdm bar is the single indicator
                chunk_kw["progress"] = False

                args = [(i, chunk_kw, states[i]) for i in range(self.nchains)]

                if self.ncores > 1:
                    chunk_results = pool.map(self._run_single_with_state, args)
                else:
                    chunk_results = [self._run_single_with_state(a) for a in args]

                for i, (chain, log_prob, final_state) in enumerate(chunk_results):
                    all_chains[i].append(chain)
                    all_log_probs[i].append(log_prob)
                    states[i] = final_state

                total_steps += check_every
                iteration   += 1

                merged = MultiChainResults(
                    chains    = [np.vstack(c)  for c in all_chains],
                    log_probs = [np.concatenate(lp) for lp in all_log_probs],
                    param_names = self.sampler_kwargs["pm"].free_names
                )

                rhat_max = float(np.max(merged.rhat))
                ess_min_val = float(np.min(merged.ess))
                bar.update(check_every)
                bar.set_postfix_str(
                    f"chains: {self.nchains} | max R\u0302: {rhat_max:.4f} | "
                    f"min ESS: {ess_min_val:.0f} | "
                    f"status: R\u0302<{1+rhat_tol:.4f} {'✓' if rhat_max - 1 < rhat_tol else '✗'}",
                    refresh=True,
                )

                try:
                    if merged.is_converged(tol=rhat_tol, ess_min=ess_min, tau_factor=tau_factor):
                        bar.set_postfix_str(
                            f"chains: {self.nchains} | max R\u0302: {rhat_max:.4f} | "
                            f"min ESS: {ess_min_val:.0f} | "
                            f"status: converged \u2713",
                            refresh=True,
                        )
                        bar.close()
                        return merged, {"status": "converged",
                                        "total_steps": total_steps,
                                        "iterations": iteration}
                except Exception:
                    pass  # too few samples for reliable diagnostics — keep running

        bar.close()
        return merged, {"status": "max_steps_reached",
                        "total_steps": total_steps,
                        "iterations": iteration}


# ---------------------------------------------------------------------------
# Minimal context-manager shim so "with _NullPool() as pool" works when
# ncores == 1 (avoids spawning any subprocesses).
# ---------------------------------------------------------------------------
class _NullPool:
    """Drop-in replacement for multiprocessing.Pool when ncores == 1."""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def map(self, fn, iterable):
        return [fn(x) for x in iterable]

