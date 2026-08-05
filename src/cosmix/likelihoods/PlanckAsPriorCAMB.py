"""PlanckAsPriorCAMB — Planck primordial-amplitude prior, computed per sample.

This is the corrected replacement for :class:`PlanckAsPrior`.  Both impose the
Planck 2018 constraint

    ln(10^10 A_s) = 3.044 +/- 0.014        [arXiv:1807.06209, Table 1]

but they differ in how ``A_s`` is inferred from the sampled ``sigma80``.

Why the old prior had to be replaced
------------------------------------
``PlanckAsPrior.lnlike`` evaluates

    lnAs_implied = lnAs_mean + 2 ln(sigma80 / sigma80_ref)
    chi2         = ((lnAs_implied - lnAs_mean) / lnAs_sigma)^2

in which ``lnAs_mean`` cancels identically, leaving

    chi2 = (2 ln(sigma80 / sigma80_ref) / lnAs_sigma)^2 .

So the Planck central value never enters the arithmetic: the constraint is a
0.7%-wide Gaussian pulling ``sigma80`` toward a hand-supplied
``sigma80_ref``.  When that reference is taken from a companion LCDM fit to
the *same* data — as the class docstring instructs — the prior asks the chain
to agree with itself, and "sigma80 returns to baseline" is guaranteed by
construction rather than by agreement with Planck.

The physics this class restores
-------------------------------
The present-day amplitude factorises as

    sigma80^2 = A_s * J(theta) * D0^2                                    (1)

where ``J`` is the transfer-function/window integral (a function of
Omega_m h^2, Omega_b h^2, n_s and h) and ``D0`` is the *total* growth from the
transfer epoch to today.  Note ``D0`` is not the ``D(z)/D(0)`` used for
fsigma8 — that normalisation divides out precisely the factor needed here.

A Boltzmann code run for LCDM at a fixed reference amplitude gives

    sigma8_ref(theta)^2 = A_s_ref * J(theta) * D0_LCDM^2 .               (2)

Dividing (1) by (2) cancels ``J`` exactly and leaves

    A_s_implied = A_s_ref * (sigma80 / sigma8_ref)^2 / R^2,   R = D0_mod/D0_LCDM

    ln(10^10 A_s_implied) = ln(10^10 A_s_ref)
                            + 2 ln(sigma80 / sigma8_ref(theta))
                            - 2 ln R(theta) .                            (3)

Both corrections are absent from the old prior, which effectively sets the
J-ratio and ``R`` to unity:

* ``sigma8_ref(theta)`` — per-sample, from a cached CAMB table, so the mapping
  responds to each sample's own Omega_m0, H0, omega_b and n_s.
* ``R(theta)`` — from :meth:`GrowthKinematics.growth_ratio`.  Weaker gravity
  suppresses growth (R < 1), so a *larger* primordial amplitude is needed to
  reach the same sigma80 today; this term dominates for lambda0 != 0.

Cached sigma8_ref table
-----------------------
Running CAMB inside the likelihood is impossible (~315 ms per call against
~10^5-10^6 likelihood evaluations), so ``sigma8_ref`` is interpolated from a
4D table over (Omega_m0, H0, omega_b, n_s), built once by the companion
grid-builder script and stored as ``sigma8_ref_grid_wide.npz``.
Interpolation is done on log(sigma8) — smoother, and positive by
construction.

Outside the tabulated box the likelihood returns -inf.  The box is chosen
wide enough that the geometric likelihoods (compressed CMB, BAO, SNe) have
already driven the total likelihood to negligible values there, so this acts
as a harmless bound rather than a physical prior cut; ``n_offgrid`` counts
such evaluations so the assumption can be audited after a run.

Usage
-----
In input.yaml::

    likelihoods:
      - name: PlkAsCAMB
        class: PlanckAsPriorCAMB
        options:
          grid_file: null      # defaults to sigma8_ref_grid_wide.npz

Unlike the old prior this takes no ``sigma80_ref`` — there is nothing to
calibrate, which is the entire point.

References
----------
- Planck 2018 results VI, A&A 641 A6 (2020); arXiv:1807.06209
- Eisenstein & Hu (1998), ApJ 496:605  (transfer-function context)
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from cosmix.core.Registry import cosmix_registry
from cosmix.core.LikelihoodBase_ import LikelihoodBase
from cosmix.core.ParameterManager_ import Parameter, UniformPrior
from cosmix.core.Paths import DATA_DIR

_DEFAULT_GRID = DATA_DIR / "sigma8_ref" / "sigma8_ref_grid_wide.npz"

# Single redshift at which growth_ratio is requested. R is a z=0 scalar; the
# cache is keyed on a redshift grid, so we ask for it at one point.
_Z_PROBE = np.array([0.0])


@cosmix_registry.register_likelihood("planckaspriorcamb")
class PlanckAsPriorCAMB(LikelihoodBase):
    """Gaussian prior on ln(10^10 A_s), inferred per sample.

    Parameters
    ----------
    pm : ParameterManager
    grid_file : str or Path, optional
        Cached CAMB sigma8_ref table.  Defaults to
        ``nbks_n_others/sigma8_ref_grid_wide.npz``.
    lnAs_mean, lnAs_sigma : float
        Planck 2018 constraint on ln(10^10 A_s).  Defaults 3.044, 0.014.
    """

    name = "PlkAsCAMB"

    def __init__(self, pm, grid_file=None, lnAs_mean=3.044, lnAs_sigma=0.014):
        super().__init__(pm)

        path = Path(grid_file) if grid_file is not None else _DEFAULT_GRID
        if not path.exists():
            raise FileNotFoundError(
                f"[PlanckAsPriorCAMB] sigma8_ref table not found at {path}. "
                f"Build it first with the grid-builder script (a few thousand "
                f"CAMB calls, ~30 min, done once)."
            )

        d = np.load(path)
        self._grid_path = path
        self.As_ref = float(d["As_ref"])
        self._interp = RegularGridInterpolator(
            (d["Om_ax"], d["H0_ax"], d["wb_ax"], d["ns_ax"]),
            np.log(d["grid"]),
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        self._box = {
            "Omegam0": (float(d["Om_ax"][0]), float(d["Om_ax"][-1])),
            "H0":      (float(d["H0_ax"][0]), float(d["H0_ax"][-1])),
            "omega_b": (float(d["wb_ax"][0]), float(d["wb_ax"][-1])),
            "n_s":     (float(d["ns_ax"][0]), float(d["ns_ax"][-1])),
        }

        self.lnAs_mean = lnAs_mean
        self.lnAs_sigma = lnAs_sigma
        self.data_size = 1
        self.produce_residuals = False

        #: Evaluations rejected for falling outside the tabulated box.
        self.n_offgrid = 0

    @classmethod
    def declare_parameters(cls):
        """Own sigma80, mirroring PlanckAsPrior so either can supply it.

        Other likelihoods that also use sigma80 (RSD, SDSSDR16, ...) rely on
        Pipeline deduplication rather than re-declaring it.
        """
        return [
            Parameter(
                name="sigma80",
                latex=r"\sigma_{80}",
                prior=UniformPrior(0.6, 1.2),
                role="nuisance",
                status="free",
                value=0.8,
                proposed_scale=0.01,
            )
        ]

    def get_requirements(self):
        """Request the total-growth ratio R = D0_model / D0_LCDM.

        This is what couples the prior to the modified growth sector; the old
        prior requested nothing at all and so could not see it.
        """
        return {"growth_ratio": _Z_PROBE}

    def sigma8_ref(self, theta):
        """CAMB sigma8(z=0) at the fixed reference amplitude A_s_ref.

        Returns NaN outside the tabulated box.
        """
        H0 = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        omega_b = self.pm.get_value(theta, "omega_b")
        n_s = self.pm.get_value(theta, "n_s")
        return float(np.exp(self._interp([[Omegam0, H0, omega_b, n_s]])[0]))

    def lnlike(self, theta, theory):
        sigma80 = self.pm.get_value(theta, "sigma80")
        if sigma80 <= 0.0:
            return -np.inf

        s8ref = self.sigma8_ref(theta)
        if not np.isfinite(s8ref) or s8ref <= 0.0:
            # Outside the tabulated box -- see class docstring.
            self.n_offgrid += 1
            return -np.inf

        R = float(np.atleast_1d(theory.eval("growth_ratio", _Z_PROBE))[0])
        if not np.isfinite(R) or R <= 0.0:
            return -np.inf

        # Eq. (3) of the module docstring.
        lnAs_implied = (np.log(1e10 * self.As_ref)
                        + 2.0 * np.log(sigma80 / s8ref)
                        - 2.0 * np.log(R))

        chi2 = ((lnAs_implied - self.lnAs_mean) / self.lnAs_sigma) ** 2
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2

    def norm_term(self):
        return 0.0

    def implied_lnAs(self, theta, theory):
        """ln(10^10 A_s_implied) for this sample — the diagnostic itself.

        Same quantity ``lnlike`` constrains, exposed for post-processing so
        chains can be summarised without re-deriving Eq. (3).
        """
        sigma80 = self.pm.get_value(theta, "sigma80")
        s8ref = self.sigma8_ref(theta)
        R = float(np.atleast_1d(theory.eval("growth_ratio", _Z_PROBE))[0])
        if not (np.isfinite(s8ref) and np.isfinite(R)) or min(s8ref, R, sigma80) <= 0:
            return np.nan
        return (np.log(1e10 * self.As_ref)
                + 2.0 * np.log(sigma80 / s8ref) - 2.0 * np.log(R))
