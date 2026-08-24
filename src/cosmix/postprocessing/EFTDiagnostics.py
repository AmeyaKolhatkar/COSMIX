"""EFTDiagnostics — strong-coupling proximity diagnostics for modified-gravity posteriors.

Computes the dimensionless ratio

    eps_sc(z) = f_QQ(Q) * Q / f_Q(Q)                                      (1)

over posterior samples.  Eq. (1) is defined for *any* Lagrangian f of a single
curvature-like scalar, so the machinery here is written against the
derivatives themselves and knows nothing about how any particular model is
parameterised: supply ``f_Q``, ``f_QQ`` and ``Q_of_z`` as callables and
everything downstream -- posterior evaluation, weighted intervals, the
worst-case scan over redshift, plotting, archiving -- follows.

Nothing in the core section below refers to a parameter name, a density
parameter, or a background form.  All of that lives either in the caller's
callables or in a :class:`ModelSpec` registered for a given archived model
name.  The specs for the models shipped with this package are registered in
the clearly-marked BUILT-IN section at the end of the file, and are in no way
privileged: they use only the public API and could be deleted or moved to
another module without touching the core.

Three ways to build one
-----------------------
1. **Raw callables** -- any f(Q), nothing registered::

       EFTDiagnostics(f_Q=lambda Q: 1 + 2*Q/M2,
                      f_QQ=lambda Q: np.full_like(Q, 2/M2),
                      Q_of_z=lambda z: -6*H0**2*E(z)**2,
                      weights=w, label="quadratic")

2. **From a model object** -- anything implementing the ``f_prime`` /
   ``f_double_prime`` protocol of
   :class:`cosmix.theory.CurvedfQBase.CurvedfQBase`::

       EFTDiagnostics.from_model(my_model, Q_of_z=...)

3. **From an archived run** -- registry dispatch on the recorded model name::

       EFTDiagnostics.from_run("runs/.../run_Base_SDSS_fQ_lambda0")

To teach it a new model, register a spec; do not edit this module::

    register_model(ModelSpec(
        name="MyModel",                     # matches manifest model.name
        params=("H0", "Omegam0", "beta"),   # pulled from chain or manifest
        build=my_builder,                   # params -> (f_Q, f_QQ, Q_of_z)
    ))

The companion diagnostic: quasi-static validity
-----------------------------------------------
The scale-independent coupling mu_G = 1/f_Q used in the RSD and E_G
likelihoods follows from the quasi-static approximation (QSA), which retains
the k^2/a^2 gradient terms and drops those lacking that enhancement.  Any
dropped term is suppressed relative to what is kept by

    (aH/ck)^2 ,        N = ln a                                           (2)

which is a statement about scales only.  An earlier version of this module
weighted Eq. (2) by [1 + |d ln f_Q/dN|], reasoning that terms carrying time
derivatives of f_Q enter the linearised equations as they do in f(R) and
Horndeski.  That analogy was not verified against the f(Q) perturbation
equations of arXiv:1906.10027 Sec. IV.E and no such term was found there;
it is also structurally doubtful, since the coincident gauge trivialises the
connection and leaves second-order field equations.  The weight is gone.
What remains model-specific, and is directly checkable because it follows
from mu_G alone, is the coupling's running rate: since Q is proportional to
H^2 in flat FLRW, (dQ/dN)/Q = 2 dlnH/dN exactly, so

    d ln f_Q/dN = 2 eps_sc (dlnH/dN) = -2 (1+q) eps_sc                      (3)

with q the deceleration parameter -- so the running rate is eps_sc times a
purely background factor, and the two diagnostics remain one argument rather
than two.  :meth:`EFTDiagnostics.dln_fQ_dN` evaluates the left-hand side of
(3) numerically from ``f_Q(Q_of_z(z))``, making no use of the identity, so the
identity itself stays available as a check;
:meth:`EFTDiagnostics.dln_muG_dN` returns its negative, the running rate of
the coupling.  :meth:`EFTDiagnostics.subhorizon_ratio` gives Eq. (2).

Note the division of labour: Eq. (2) says whether the modes are deep enough
inside the horizon for a quasi-static treatment at all, Eq. (3) says whether
the coupling is slow enough not to stress it, and eps_sc says how close the
theory sits to the boundary where the treatment stops being meaningful.  None
of the three is a substitute for a full perturbative analysis.

Because k is conventionally quoted in h/Mpc, H0 cancels from (aH/ck)^2 and
the only background input is E(z) = H(z)/H0, which is recovered from
``Q_of_z`` as sqrt(Q(z)/Q(0)).  Pass ``E_of_z`` explicitly if your Q is not
proportional to H^2.

What eps_sc is, and what it is not
----------------------------------
Eq. (1) is *algebraically exact*: the ratio of the second to the first
derivative of f, made dimensionless by Q.  It is the natural parameter
controlling how far the theory sits from its General-Relativity limit
(eps_sc -> 0 as f_QQ -> 0).

Its reading as *strong-coupling proximity* is heuristic.  eps_sc is not the
strong-coupling scale Lambda_sc, and it does not diagnose the kinetic-matrix
degeneracy responsible for the pathologies reported for f(Q) on cosmological
backgrounds (arXiv:2311.04201).  Small eps_sc is a *necessary but not
sufficient* condition for the effective description to be trustworthy, and
should be quoted alongside an explicit EFT-below-a-cutoff framing rather than
as a proof of health.  Note too that eps_sc -> 0 *is* the GR limit, so "eps_sc
is small" partly restates "the model is close to GR".

The defensible claim it supports is bounded: *the data do not drive the model
into the regime where the higher-order structure of f dominates, at any
redshift where the theory is used.*

References
----------
- arXiv:2311.04201  strong coupling / ghosts in f(Q) cosmology
- arXiv:2302.03545  analogous EFT-with-cutoff treatment in f(T)
"""

import json
import numpy as np
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = [
    # ── generic core ────────────────────────────────────────────────────────
    "EFTDiagnostics", "ModelSpec", "register_model", "registered_models",
    "get_model_spec", "weighted_quantile", "plot_epsilon_sc",
    "Z_RECOMBINATION", "DEFAULT_Z",
    # ── built-in f(Q) family ────────────────────────────────────────────────
    "closed_form_eps0", "alpha3_from_flatness", "E_of_z",
    "verify_against_model",
]

#: Photon-decoupling redshift, a convenient reference point for the
#: high-redshift end of the diagnostic.  Not used by the core machinery.
Z_RECOMBINATION = 1090.0

#: Default redshift grid: linear across the typical data range, log beyond.
DEFAULT_Z = np.concatenate([
    np.linspace(0.0, 3.0, 61),
    np.logspace(np.log10(3.0), np.log10(Z_RECOMBINATION), 60)[1:],
])

#: c / (100 km/s/Mpc), in Mpc/h.  Converts aH/(ck) when k is given in h/Mpc,
#: in which units H0 cancels from the ratio entirely.
C_OVER_H100 = 2997.92458

#: Representative wavenumber for RSD / E_G measurements, h/Mpc.
K_RSD = 0.1

#: Redshift range over which the quasi-static error is assessed -- the span
#: of the growth and BAO data, NOT the full DEFAULT_Z grid.  (aH/ck)^2 *grows*
#: with redshift, so a maximum taken out to recombination would be dominated
#: by z ~ 1000, where a k ~ 0.1 h/Mpc mode is not sub-horizon and the
#: quasi-static question does not arise.
Z_GROWTH = np.linspace(0.0, 2.5, 201)


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC CORE -- no knowledge of any model, parameter name, or background
# ══════════════════════════════════════════════════════════════════════════════
def weighted_quantile(values, weights, quantiles):
    """Weighted quantiles, ignoring non-finite samples."""
    values = np.asarray(values, dtype=float)
    weights = (np.ones_like(values) if weights is None
               else np.asarray(weights, dtype=float))
    ok = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[ok], weights[ok]
    if values.size == 0:
        return np.full(np.shape(quantiles), np.nan)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return np.interp(quantiles, cdf, values)


@dataclass(frozen=True)
class ModelSpec:
    """How to rebuild f-derivatives for one archived model.

    Attributes
    ----------
    name : str
        Must match ``manifest["model"]["name"]``.
    params : tuple of str
        Parameter names this model needs.  :meth:`EFTDiagnostics.from_run`
        resolves each from the chain, else the manifest's recorded fixed
        value, else `defaults`.
    build : callable
        ``build(params) -> (f_Q, f_QQ, Q_of_z)``, where `params` maps each
        name in `params` to a per-sample array of shape (n_sample,).  The
        returned callables receive Q of shape (n_sample, n_z) and must
        broadcast against it -- close over parameters shaped (n_sample, 1).
        ``Q_of_z(z)`` returns Q on the same (n_sample, n_z) grid.
    defaults : dict, optional
        Fallbacks for parameters absent from both chain and manifest, e.g.
        a coefficient a run held fixed without recording it.
    closed_form : callable, optional
        ``closed_form(params) -> eps_sc(z=0)`` by an independent route, used
        by :meth:`EFTDiagnostics.check_closed_form` as a cross-check.
    description : str, optional
        Free text carried into archived summaries.
    """
    name: str
    params: tuple
    build: Callable
    defaults: dict = field(default_factory=dict)
    closed_form: Callable = None
    description: str = ""


_REGISTRY: dict = {}


def register_model(spec: ModelSpec, overwrite: bool = False):
    """Register a :class:`ModelSpec` so ``from_run`` can dispatch on it."""
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"{spec.name!r} is already registered; pass overwrite=True to replace it."
        )
    _REGISTRY[spec.name] = spec
    return spec


def registered_models():
    """Names that :meth:`EFTDiagnostics.from_run` can reconstruct."""
    return sorted(_REGISTRY)


def get_model_spec(name):
    """Return the registered :class:`ModelSpec` for `name`."""
    if name not in _REGISTRY:
        raise KeyError(
            f"No ModelSpec registered for {name!r}. Known: {registered_models()}. "
            f"Use register_model() to add one."
        )
    return _REGISTRY[name]


class EFTDiagnostics:
    """Strong-coupling diagnostics for a modified-gravity posterior.

    Parameters
    ----------
    f_Q, f_QQ : callable
        First and second derivatives of f with respect to Q.  Called with Q
        of shape (n_sample, n_z); must broadcast against per-sample
        parameters.  ``f_QQ`` may return exact zeros (the GR limit), which
        yields eps_sc = 0 without any special-casing.
    Q_of_z : callable
        ``z -> Q`` of shape (n_sample, n_z).  For flat FLRW in f(Q) gravity
        this is Q = -6 H(z)^2, but the core imposes no such assumption.
    f : callable, optional
        f(Q) itself.  Unused by eps_sc; carried for callers wanting other
        combinations.
    weights : array_like, optional
        Posterior weights (n_sample,).  Uniform if omitted.
    label : str, optional
        Used in figure legends and archived summaries.
    meta : dict, optional
        Free-form provenance recorded by :meth:`save`.
    """

    def __init__(self, f_Q, f_QQ, Q_of_z, *, f=None, weights=None,
                 label=None, meta=None):
        self.f_Q, self.f_QQ, self.Q_of_z, self.f = f_Q, f_QQ, Q_of_z, f
        self.weights = None if weights is None else (
            np.asarray(weights, dtype=float) / np.sum(weights))
        self.label = label
        self.meta = dict(meta or {})
        self._params, self._spec = None, None

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_model(cls, model, Q_of_z, **kw):
        """Build from any object exposing the ``CurvedfQBase`` protocol.

        Requires ``f_prime`` and ``f_double_prime``; ``f`` is used when
        present.  This is the hook for models defined outside this package:
        implement those methods and the diagnostics follow.
        """
        for attr in ("f_prime", "f_double_prime"):
            if not callable(getattr(model, attr, None)):
                raise TypeError(
                    f"{type(model).__name__} does not implement {attr}(); the "
                    f"CurvedfQBase protocol (f, f_prime, f_double_prime) is "
                    f"required by from_model. Pass callables to the "
                    f"constructor directly if the model has another shape."
                )
        kw.setdefault("label", getattr(model, "name", None))
        return cls(model.f_prime, model.f_double_prime, Q_of_z,
                   f=getattr(model, "f", None), **kw)

    @classmethod
    def from_run(cls, run_dir, spec=None, **kw):
        """Reconstruct from an archived run directory.

        Reads ``manifest.yaml`` / ``chain.npy`` / ``weights.npy`` (the
        read-side counterpart of ``RunArchive``) and dispatches on the
        recorded model name, unless `spec` is given explicitly.

        Each parameter the spec declares is resolved in order from: the
        sampled chain, the manifest's recorded fixed value, then the spec's
        `defaults`.  A run that held a parameter fixed therefore needs no
        special-casing by the caller.
        """
        run_dir = Path(run_dir)
        with open(run_dir / "manifest.yaml") as fh:
            manifest = yaml.safe_load(fh)
        name = manifest["model"]["name"]
        spec = spec or get_model_spec(name)

        names = manifest["labels"]["names"]
        chain = np.load(run_dir / "chain.npy")
        w_path = run_dir / "weights.npy"
        weights = np.load(w_path) if w_path.exists() else None

        params = {}
        for key in spec.params:
            if key in names:
                params[key] = chain[:, names.index(key)]
                continue
            entry = manifest.get("parameters", {}).get(key) or {}
            value = entry.get("value", spec.defaults.get(key))
            if value is None:
                raise KeyError(
                    f"{run_dir.name}: {key!r} is required by ModelSpec "
                    f"{spec.name!r} but is neither sampled, nor fixed with a "
                    f"recorded value in the manifest, nor supplied via "
                    f"ModelSpec.defaults."
                )
            params[key] = np.full(len(chain), float(value))

        f_Q, f_QQ, Q_of_z = spec.build(params)
        kw.setdefault("label", run_dir.name)
        obj = cls(f_Q, f_QQ, Q_of_z, weights=weights, **kw)
        obj.meta.update(model=name, run_dir=str(run_dir), n_samples=len(chain),
                        description=spec.description)
        obj._params, obj._spec = params, spec
        return obj

    # ── diagnostics ─────────────────────────────────────────────────────────
    def epsilon_sc(self, z=None):
        """eps_sc(z) = f_QQ Q / f_Q on an (n_sample, n_z) grid."""
        z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))
        Q = self.Q_of_z(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.f_QQ(Q) * Q / self.f_Q(Q)

    def worst_case(self, z=None):
        """Largest |eps_sc| over `z`, per sample, and where it occurs.

        Uses an explicit scan rather than assuming the maximum falls at z = 0:
        when f_QQ carries terms of opposite sign that cancel at some
        redshift, the extremum can sit in the interior of the range.
        """
        z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))
        a = np.abs(self.epsilon_sc(z))
        finite = np.isfinite(a)
        i = np.argmax(np.where(finite, a, -np.inf), axis=1)
        best = a[np.arange(len(a)), i]
        allbad = ~finite.any(axis=1)
        return np.where(allbad, np.nan, best), z[i]

    # ── quasi-static validity (Eq. 2) ───────────────────────────────────────
    def dln_fQ_dN(self, z=None, h=1e-4):
        """d ln f_Q / dN with N = ln a, by central difference in ln a.

        Evaluated numerically from ``f_Q(Q_of_z(z))`` rather than via the
        identity d ln f_Q/dN = 2 eps_sc dlnH/dN, so that the identity stays
        available as an independent check (see :func:`verify_against_model`).

        Differences are taken on f_Q itself rather than on log f_Q, so a
        negative f_Q -- unphysical, but reachable in the tails of a prior --
        yields a real result instead of a NaN.
        """
        z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))
        one_plus_z = 1.0 + z
        # a -> a e^{+h} corresponds to (1+z) -> (1+z) e^{-h}
        z_plus = one_plus_z * np.exp(-h) - 1.0
        z_minus = one_plus_z * np.exp(+h) - 1.0
        f_plus = self.f_Q(self.Q_of_z(z_plus))
        f_minus = self.f_Q(self.Q_of_z(z_minus))
        f_here = self.f_Q(self.Q_of_z(z))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (f_plus - f_minus) / (2.0 * h) / f_here

    def subhorizon_ratio(self, z=None, k=K_RSD, E_of_z=None):
        """(aH/ck)^2 -- the standard sub-horizon expansion parameter.

        The quasi-static limit retains the k^2-enhanced gradient term and
        drops the rest; anything lacking that enhancement is suppressed by
        this factor, so it sets the scale of the neglected terms.  It is a
        statement about scales alone and carries no model-specific content
        beyond the background H(z) -- which, for a background-inert coupling,
        is essentially common to all the models here.

        This deliberately does *not* include any coupling-running weight.  An
        earlier version of this module multiplied by [1 + |d ln f_Q/dN|] on
        the grounds that terms carrying time derivatives of f_Q appear in the
        linearised equations, by analogy with the alpha_M term of f(R) and
        Horndeski.  That analogy was never verified against the f(Q)
        perturbation equations of arXiv:1906.10027 Sec. IV.E, and no such
        term was found there on inspection.  It is also questionable on
        structural grounds: in the coincident gauge the connection is
        trivialised and the field equations remain second order, so f(Q) is
        not simply f(R) with M^2 = f_Q.  The weight has been removed rather
        than carried as an unsourced assumption.  See
        :meth:`dln_muG_dN` for the coupling's running rate, which is a
        property of mu_G itself and needs no perturbative input.

        Parameters
        ----------
        k : float
            Comoving wavenumber in **h/Mpc**.  In these units H0 cancels, so
            no Hubble constant is required.
        E_of_z : callable, optional
            ``z -> E = H/H0`` of shape (n_sample, n_z).  Defaults to
            sqrt(Q(z)/Q(0)), exact whenever Q is proportional to H^2.
        """
        z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))
        if E_of_z is None:
            Q_z = self.Q_of_z(z)
            Q_0 = self.Q_of_z(np.array([0.0]))
            with np.errstate(divide="ignore", invalid="ignore"):
                E = np.sqrt(Q_z / Q_0)
        else:
            E = np.asarray(E_of_z(z), dtype=float)

        aH_over_ck = E / ((1.0 + z)[None, :] * C_OVER_H100 * k)
        return aH_over_ck ** 2

    def dln_muG_dN(self, z=None, h=1e-4):
        """d ln mu_G / dN, the logarithmic running rate of the coupling.

        Since mu_G = 1/f_Q this is just -d ln f_Q/dN: a property of the
        coupling the model already defines, obtained by differentiating it.
        No perturbative input is involved, so unlike the weight removed from
        :meth:`subhorizon_ratio` this quantity is directly checkable.

        Useful as a statement that the quasi-static assumption is not being
        stressed: if the coupling varies on a Hubble time or slower
        (|d ln mu_G/dN| lesssim 1), its time derivatives are not competitive
        with the k^2/a^2 gradients that the limit retains.
        """
        return -self.dln_fQ_dN(z, h)

    def qsa_worst_case(self, z=None, k=K_RSD, **kw):
        """Largest sub-horizon ratio over `z`, per sample, and where."""
        z = Z_GROWTH if z is None else np.atleast_1d(np.asarray(z, float))
        a = self.subhorizon_ratio(z, k, **kw)
        finite = np.isfinite(a)
        i = np.argmax(np.where(finite, a, -np.inf), axis=1)
        best = a[np.arange(len(a)), i]
        return np.where(~finite.any(axis=1), np.nan, best), z[i]

    def bands(self, z=None, levels=(0.68, 0.95)):
        """Weighted median and credible bands of |eps_sc| along `z`.

        Returns ``{"z":…, "median":…, "lo68"/"hi68":…, …}``, each shape
        (n_z,).  This is what the plotter consumes and :meth:`save` archives.
        """
        z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))
        eps = np.abs(self.epsilon_sc(z))

        def q(p):
            return np.array([weighted_quantile(eps[:, j], self.weights, p)
                             for j in range(eps.shape[1])])

        out = {"z": z, "median": q(0.5)}
        for lvl in levels:
            tag = f"{round(lvl * 100)}"
            out[f"lo{tag}"], out[f"hi{tag}"] = q(0.5 - lvl / 2), q(0.5 + lvl / 2)
        return out

    def summary(self, z=None, quantiles=(0.5, 0.025, 0.975),
                extra_z=(Z_RECOMBINATION,), qsa_k=K_RSD, qsa_z=None):
        """Weighted quantiles of |eps_sc| at z=0, its worst case, and extras.

        When `qsa_k` is not None, also reports the worst-case QSA error at
        that wavenumber (h/Mpc) -- the table row the revision needs for the
        k-independence claim.  That maximum is taken over `qsa_z`, defaulting
        to :data:`Z_GROWTH` rather than the full `z` grid: see Z_GROWTH on why
        extending it to recombination would be meaningless.
        """
        z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))
        eps_max, z_at = self.worst_case(z)
        out = {
            "label": self.label,
            "quantiles": list(quantiles),
            "eps0": weighted_quantile(
                np.abs(self.epsilon_sc([0.0])[:, 0]), self.weights, quantiles),
            "eps_max": weighted_quantile(eps_max, self.weights, quantiles),
            "z_at_max": weighted_quantile(z_at, self.weights, quantiles),
        }
        if qsa_k is not None:
            zq = Z_GROWTH if qsa_z is None else np.atleast_1d(
                np.asarray(qsa_z, float))
            qsa_max, qsa_zat = self.qsa_worst_case(zq, k=qsa_k)
            run = np.abs(self.dln_muG_dN(zq))
            run_max = np.nanmax(np.where(np.isfinite(run), run, -np.inf), axis=1)
            out["qsa_k"] = qsa_k
            out["qsa_z_range"] = [float(zq[0]), float(zq[-1])]
            out["subhorizon_ratio_max"] = weighted_quantile(
                qsa_max, self.weights, quantiles)
            out["dln_muG_dN_max"] = weighted_quantile(
                run_max, self.weights, quantiles)
            out["qsa_z_at_max"] = weighted_quantile(qsa_zat, self.weights, quantiles)
        for zz in extra_z or ():
            out[f"eps_z{zz:g}"] = weighted_quantile(
                np.abs(self.epsilon_sc([zz])[:, 0]), self.weights, quantiles)
        return out

    def check_closed_form(self, rtol=1e-10):
        """Cross-check eps_sc(0) against the spec's independent closed form.

        Relative deviation where the closed form is non-zero, absolute where
        it vanishes (so exact-GR models, for which eps_sc == 0 identically,
        are handled rather than producing 0/0).
        """
        if self._spec is None or self._spec.closed_form is None:
            raise RuntimeError(
                f"No closed form registered for {getattr(self._spec, 'name', None)!r}."
            )
        general = np.atleast_1d(self.epsilon_sc([0.0])[:, 0]).astype(float)
        closed = np.broadcast_to(
            np.atleast_1d(np.asarray(self._spec.closed_form(self._params),
                                     dtype=float)), general.shape)
        ok = np.isfinite(general) & np.isfinite(closed)
        if not ok.any():
            return np.nan
        scale = np.where(np.abs(closed[ok]) > 0, np.abs(closed[ok]), 1.0)
        dev = float(np.max(np.abs(general[ok] - closed[ok]) / scale))
        assert dev < rtol, f"eps_sc(0) vs closed form: max deviation {dev:.2e}"
        return dev

    # ── output ──────────────────────────────────────────────────────────────
    def save(self, outdir, z=None, make_figure=True, store_samples=False):
        """Write summary, bands, and (optionally) the figure.

        Produces ``summary.json``, ``epsilon_sc.npz`` and ``epsilon_sc.pdf``
        under `outdir`, which is created if absent.

        The ``.npz`` holds the credible *bands* plus the per-sample worst
        case -- a few hundred kB.  The full (n_sample, n_z) eps array is
        ~17 MB for a typical nested-sampling run and recomputes from the
        chain in well under a second, so it is written only when
        ``store_samples=True`` (needed if you want to re-derive joint
        constraints rather than just replot).
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))

        payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in self.summary(z).items()}
        payload["meta"] = self.meta
        (outdir / "summary.json").write_text(json.dumps(payload, indent=2))

        eps_max, z_at = self.worst_case(z)
        arrays = dict(self.bands(z), eps_max=eps_max, z_at_max=z_at)
        if store_samples:
            arrays["eps"] = self.epsilon_sc(z)
            arrays["weights"] = (self.weights if self.weights is not None
                                 else np.array([]))
        np.savez_compressed(outdir / "epsilon_sc.npz", **arrays)

        if make_figure:
            import matplotlib.pyplot as plt
            ax = plot_epsilon_sc({self.label or "posterior": self}, z=z)
            ax.figure.savefig(outdir / "epsilon_sc.pdf", bbox_inches="tight")
            plt.close(ax.figure)
        return outdir


def plot_epsilon_sc(diagnostics, out=None, z=None, levels=(0.68, 0.95),
                    show_boundary=True, ax=None, colors=None, grid=False):
    """|eps_sc|(z) posterior bands, log-log, with the eps_sc = 1 boundary.

    A bare eps_sc(z) curve carries little information -- typically it decays
    monotonically and sits far below unity.  This figure is built to make
    three points at once: the *magnitude* of eps_sc, its *decay* across the
    full range over which the theory is applied (hence the log x-axis out to
    recombination), and the *effect of a prior or dataset choice*, by
    overlaying several posteriors.

    Parameters
    ----------
    diagnostics : mapping {label: EFTDiagnostics}
        Passing two variants of one model is the intended comparison.
    """
    import matplotlib.pyplot as plt

    z = DEFAULT_Z if z is None else np.atleast_1d(np.asarray(z, float))
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6.4, 4.4))
    cycle = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, diag) in enumerate(diagnostics.items()):
        colour = cycle[i % len(cycle)]
        b = diag.bands(z, levels=levels)
        ax.plot(z, b["median"], color=colour, lw=1.8, label=label, zorder=3)
        for lvl, alpha in zip(sorted(levels, reverse=True), (0.15, 0.28)):
            tag = f"{round(lvl * 100)}"
            ax.fill_between(z, b[f"lo{tag}"], b[f"hi{tag}"],
                            color=colour, alpha=alpha, lw=0, zorder=2)

    if show_boundary:
        ax.axhline(1.0, color="0.3", ls="--", lw=1.2, zorder=1)
        ax.text(0.985, 0.85, r"$|\varepsilon_{\rm sc}| = 1$",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=9, color="0.3")

    ax.axvline(Z_RECOMBINATION, color="0.6", ls=":", lw=1.0, zorder=1)
    ax.text(Z_RECOMBINATION, 0.02, r" $z_{\rm rec}$",
            transform=ax.get_xaxis_transform(), fontsize=9,
            color="0.5", ha="left", va="bottom", rotation=90)

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$|\varepsilon_{\rm sc}(z)|$")
    ax.set_xlim(0, Z_RECOMBINATION * 1.6)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    if grid is False:
        ax.grid(visible=False)
    else:
        ax.grid(alpha=0.25, lw=0.5)

    if out is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(out, bbox_inches="tight")
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN MODELS -- specs for the theories shipped with this package.
#
# Everything below is an *application* of the public API above and is in no
# way privileged: it could be deleted, or moved to its own module, without
# the core noticing.  Use it as a template for registering your own models.
#
# The family covered is the flat member set of
#
#     f(Q) = a1 Q + a2 Q0 + a3 Q0^2 / Q + lam0 sqrt(Q Q0) ,           (2)
#
# with Q = -6 H^2 and r = Q0/Q = 1/E(z)^2 (positive, as Q and Q0 share a
# sign).  Differentiating (2),
#
#     f_Q      = a1 - a3 r^2 + (lam0/2) sqrt(r)                       (3)
#     f_QQ * Q = 2 a3 r^2    - (lam0/4) sqrt(r)                       (4)
#
# Flatness fixes a3 = (a1 - a2 - Om0 - Omr0)/3.  Because the data prefer
# a2 ~ Omega_Lambda, Omt0 = Om0 + a2 + Omr0 sits near unity and a3 is small,
# which is why the inverse-power term contributes only weakly.  At z = 0
# (r = 1), Eqs. (3)-(4) invert to the closed forms quoted in the manuscript;
# see `closed_form_eps0`.
# ══════════════════════════════════════════════════════════════════════════════
from cosmix.Constants import Omegar0  # noqa: E402  (built-in section only)


def _col(a):
    """Broadcast a per-sample array to a (n_sample, 1) column."""
    return np.atleast_1d(np.asarray(a, dtype=float))[:, None]


def alpha3_from_flatness(Omegam0, alpha2, alpha1=1.0):
    """a3 = (a1 - a2 - Om0 - Omr0)/3, the flat-space Friedmann constraint."""
    return (np.asarray(alpha1, dtype=float)
            - np.asarray(alpha2, dtype=float)
            - np.asarray(Omegam0, dtype=float) - Omegar0) / 3.0


def E_of_z(model, z, Omegam0, alpha2=None, alpha1=1.0):
    """E(z) = H(z)/H0 on an (n_sample, n_z) grid, for the built-in family.

    Mirrors ``fQLCDM._H`` / ``fQHybrid._H``.  lam0 does not appear: the
    sqrt(Q Q0) term is background-inert for this family, so H(z) is unchanged
    by it -- which is why the coupling is detectable only through
    perturbations.  Returns NaN where the Hybrid branch is unphysical,
    matching the models' own guards.
    """
    z_row = np.atleast_1d(np.asarray(z, dtype=float))[None, :]
    if model in ("LCDM", "fQ_LCDM"):
        Om0 = _col(Omegam0)
        return np.sqrt(Om0 * (1 + z_row) ** 3
                       + Omegar0 * (1 + z_row) ** 4
                       + (1.0 - Om0 - Omegar0))
    if model == "fQ_Hybrid":
        if alpha2 is None:
            raise ValueError("fQ_Hybrid requires alpha2.")
        Om0, a2, a1 = _col(Omegam0), _col(alpha2), _col(alpha1)
        wt = Om0 * (1 + z_row) ** 3 + a2 + Omegar0 * (1 + z_row) ** 4
        wt0 = Om0 + a2 + Omegar0
        disc = np.where((d := wt ** 2 + 4 * a1 * (a1 - wt0)) < 0, np.nan, d)
        arg = (wt + np.sqrt(disc)) / (2 * a1)
        return np.sqrt(np.where(arg < 0, np.nan, arg))
    raise ValueError(f"E_of_z has no built-in background for {model!r}.")


def closed_form_eps0(model, Omegam0, lambda0, alpha2=None, alpha1=1.0):
    """eps_sc(z=0) from the manuscript's inverted closed forms.

    Independent of the general Eq. (1) path (no Q(z) evaluation), so
    agreement between the two is a genuine check on both.
    """
    if model == "LCDM":
        return np.zeros_like(np.asarray(Omegam0, dtype=float))
    lam = np.asarray(lambda0, dtype=float)
    if model == "fQ_LCDM":
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / (-2.0 - 4.0 / lam)      # -> 0 as lam -> 0
    Omt0 = (np.asarray(Omegam0, dtype=float)
            + np.asarray(alpha2, dtype=float) + Omegar0)
    num = alpha1 - (alpha1 - Omt0) / 3.0 + lam / 2.0
    den = (2.0 / 3.0) * (alpha1 - Omt0) - lam / 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return den / num


def _build_gr(params):
    """f(Q) = Q + const: f_Q = 1, f_QQ = 0, hence eps_sc == 0 exactly.

    Registered so that GR baselines can be run through the same pipeline as
    the modified models -- useful as a null control in comparison figures
    and tables -- without the caller special-casing them.
    """
    def f_Q(Q):
        return np.ones_like(Q)

    def f_QQ(Q):
        return np.zeros_like(Q)

    def Q_of_z(z):
        E = E_of_z("LCDM", z, params["Omegam0"])
        return -6.0 * _col(params["H0"]) ** 2 * E ** 2

    return f_Q, f_QQ, Q_of_z


def _build_polynomial(has_alpha3):
    """Builder factory for the flat members of Eq. (2)."""
    model = "fQ_Hybrid" if has_alpha3 else "fQ_LCDM"

    def build(params):
        a1, lam = _col(params["alpha1"]), _col(params["lambda0"])
        Q0 = -6.0 * _col(params["H0"]) ** 2
        a3 = (_col(alpha3_from_flatness(params["Omegam0"], params["alpha2"],
                                        params["alpha1"]))
              if has_alpha3 else 0.0)

        def f_Q(Q):                                   # Eq. (3)
            r = Q0 / Q
            return a1 - a3 * r ** 2 + 0.5 * lam * np.sqrt(r)

        def f_QQ(Q):                                  # Eq. (4), divided by Q
            r = Q0 / Q
            return (2.0 * a3 * r ** 2 - 0.25 * lam * np.sqrt(r)) / Q

        def Q_of_z(z):
            E = E_of_z(model, z, params["Omegam0"],
                       params.get("alpha2") if has_alpha3 else None,
                       params["alpha1"])
            return Q0 * E ** 2

        return f_Q, f_QQ, Q_of_z
    return build


register_model(ModelSpec(
    name="LCDM",
    params=("H0", "Omegam0"),
    build=_build_gr,
    closed_form=lambda p: closed_form_eps0("LCDM", p["Omegam0"], 0.0),
    description="GR baseline: f(Q) = Q + const, so f_QQ = 0 and eps_sc = 0.",
))

register_model(ModelSpec(
    name="fQ_LCDM",
    params=("H0", "Omegam0", "lambda0", "alpha1"),
    build=_build_polynomial(has_alpha3=False),
    defaults={"alpha1": 1.0},
    closed_form=lambda p: closed_form_eps0(
        "fQ_LCDM", p["Omegam0"], p["lambda0"], None, p["alpha1"]),
    description="f(Q) = a1 Q + a2 Q0 + lam0 sqrt(Q Q0); background-inert coupling.",
))

register_model(ModelSpec(
    name="fQ_Hybrid",
    params=("H0", "Omegam0", "lambda0", "alpha1", "alpha2"),
    build=_build_polynomial(has_alpha3=True),
    defaults={"alpha1": 1.0},
    closed_form=lambda p: closed_form_eps0(
        "fQ_Hybrid", p["Omegam0"], p["lambda0"], p["alpha2"], p["alpha1"]),
    description="f(Q) = a1 Q + a2 Q0 + a3 Q0^2/Q + lam0 sqrt(Q Q0).",
))


def verify_against_model(model_name="fQ_Hybrid", Omegam0=0.31, alpha2=0.72,
                         lambda0=0.6, H0=68.3, z=None, rtol=1e-10):
    """Check the built-in specs against the live ``cosmix.theory`` classes.

    The f(Q) forms live in ``cosmix.theory``; re-encoding them in the
    built-in section above risks the two drifting apart.  Two comparisons
    guard against that, both treating ``cosmix.theory`` as the single source
    of truth:

    1. :func:`E_of_z` vs the model's analytic ``_H``/H0.  Compared against
       ``_H`` rather than ``BackgroundKinematics.E`` because the latter
       interpolates on a finite grid (~1e-6 relative error at the default
       150-point setting), which would mask a real discrepancy behind
       interpolation noise.
    2. Eq. (3) vs ``1/model.muG(...)`` -- the flat model classes compute
       their coupling as ``muG = 1/f_Q``, so these must agree identically.
       ``muG`` is handed this module's analytic E through a minimal stand-in
       engine, isolating the *formula* rather than re-testing the background.

    Also cross-checks the general Eq. (1) path against
    :func:`closed_form_eps0` at z = 0.

    Raises AssertionError on mismatch; returns max relative deviations.
    """
    from cosmix.core.ParameterManager_ import ParameterManager, Parameter
    from cosmix.theory.fQ_LCDM import fQLCDM
    from cosmix.theory.fQ_Hybrid import fQHybrid

    cls = {"fQ_LCDM": fQLCDM, "fQ_Hybrid": fQHybrid}[model_name]
    is_hybrid = model_name == "fQ_Hybrid"
    values = {"H0": H0, "Omegam0": Omegam0, "lambda0": lambda0,
              "alpha1": 1.0, "alpha2": alpha2}

    pm = ParameterManager()
    for p in cls.declare_parameters():
        pm.add_parameter(Parameter(name=p.name, latex=p.latex, prior=p.prior,
                                   role=p.role, status="fixed",
                                   value=values.get(p.name, p.value)))
    pm.freeze()
    model, theta = cls(pm), np.array([])
    z = np.linspace(0.0, 3.0, 61) if z is None else np.atleast_1d(z)

    # 1. E(z) against the model's analytic H(z).
    E_model = (model._H(z, H0, Omegam0, 1.0, alpha2, lambda0) / H0 if is_hybrid
               else model._H(z, H0, Omegam0) / H0)
    E_here = E_of_z(model_name, z, [Omegam0], [alpha2] if is_hybrid else None)[0]
    dE = np.max(np.abs(E_here / E_model - 1.0))
    assert dE < rtol, f"E(z) mismatch vs {model_name}._H: max rel dev {dE:.2e}"

    # 2. f_Q against 1/muG, with muG evaluated on this module's own E.
    class _EOnly:
        """Minimal stand-in: model.muG only ever calls bg_engine.E(z)."""
        def E(self, zz):
            return E_of_z(model_name, zz, [Omegam0],
                          [alpha2] if is_hybrid else None)[0]

    spec = get_model_spec(model_name)
    params = {k: np.array([values[k]]) for k in spec.params}
    f_Q, f_QQ, Q_of_z = spec.build(params)

    fQ_model = 1.0 / model.muG(z, theta, _EOnly())
    dF = np.max(np.abs(f_Q(Q_of_z(z))[0] / fQ_model - 1.0))
    assert dF < rtol, f"f_Q mismatch vs 1/muG: max rel dev {dF:.2e}"

    # 3. General path vs the closed form at z = 0.
    Q0 = Q_of_z([0.0])
    general = float((f_QQ(Q0) * Q0 / f_Q(Q0))[0, 0])
    closed = float(np.atleast_1d(spec.closed_form(params))[0])
    dC = abs(general / closed - 1.0) if closed != 0 else abs(general - closed)
    assert dC < rtol, f"eps_sc(0) vs closed form: deviation {dC:.2e}"

    # 4. Numerical d ln f_Q/dN against the analytic identity, Eq. (3):
    #    d ln f_Q/dN = 2 eps_sc dlnH/dN.  qsa_error() deliberately uses the
    #    numerical route, so this is a genuine independent check of it.
    diag = EFTDiagnostics(f_Q, f_QQ, Q_of_z)
    z_id = np.array([0.0, 0.5, 1.0, 2.0])
    numeric = diag.dln_fQ_dN(z_id)[0]
    hh = 1e-5
    E_p = E_of_z(model_name, (1 + z_id) * np.exp(-hh) - 1, [Omegam0],
                 [alpha2] if is_hybrid else None)[0]
    E_m = E_of_z(model_name, (1 + z_id) * np.exp(+hh) - 1, [Omegam0],
                 [alpha2] if is_hybrid else None)[0]
    dlnH_dN = (np.log(E_p) - np.log(E_m)) / (2 * hh)
    analytic = 2.0 * diag.epsilon_sc(z_id)[0] * dlnH_dN
    scale = np.maximum(np.abs(analytic), 1e-12)
    dI = float(np.max(np.abs(numeric - analytic) / scale))
    assert dI < 1e-6, (
        f"d ln f_Q/dN numeric vs identity 2 eps_sc dlnH/dN: max rel dev "
        f"{dI:.2e} (finite-difference limited; loosen only if h changed)"
    )

    return {"dE": dE, "d_fQ": dF, "d_closed_form": dC, "d_identity": dI}
