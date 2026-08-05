"""CAMBKinematics — CAMB / MGCAMB wrapper as an ObservableEngineBase engine.

Provides:
    Pk_linear(k, z)     → 2D array, shape (nk, nz), units (Mpc/h)³
    Pk_nonlinear(k, z)  → same, with halofit non-linear corrections
    sigma8_camb(z)      → 1D array (validation cross-check vs GrowthKinematics)
    r_drag(z)           → sound horizon at drag epoch in Mpc/h (z unused, API compat)

Boltzmann backend selection (automatic):
    Tries `import mgcamb` first (preferred for modified gravity science).
    Falls back to `import camb` (sufficient for ΛCDM and background-inert models).
    Install via:  pip install mgcamb   OR   pip install camb

Parameter mapping from COSMIX theta:
    H0       [km/s/Mpc]
    Omegam0  total matter density (baryon + CDM)
    Omegab0  baryon density (fixed parameter in most models, ~0.0493)
    sigma80  σ₈ today (free or fixed nuisance)

sigma80 normalisation:
    CAMB normalises P(k,z) via A_s. CAMBKinematics runs CAMB at a fixed
    reference A_s (Planck 2018 best-fit), retrieves sigma8_ref = sigma8(z=0),
    and rescales all returned P(k,z) by (sigma80 / sigma8_ref)².
    This keeps sigma80 as the sampled parameter without re-running CAMB.

MGCAMB modified gravity (MGCAMBKinematics):
    MGCAMB uses the set_mgparams() API on CAMBparams. It supports built-in
    parametric models only — there is NO API for arbitrary callable μ(k,z).
    Supported schemes: μ/Σ (mu0, sigma0), BZ (B1/B2/lambda), Linder gamma,
    f(R) (F_R0/FRn), custom 11-point spline (MGCAMB_Mu_idx_1..11).

    Models using MGCAMBKinematics must implement:
        mg_camb_params(theta, bg_engine) → dict
    returning kwargs passed directly to pars.set_mgparams().

    For background-inert scale-independent models (fQ_LCDM family), use
    MGCAMBWrapper instead — it is exact and requires only plain camb.
"""

import numpy as np
from cosmix.core.ObservableEngineBase import ObservableEngineBase
from scipy.interpolate import RegularGridInterpolator

# Prefer MGCAMB; fall back to plain CAMB transparently.
try:
    import mgcamb as camb
    _CAMB_AVAILABLE = True
    _USING_MGCAMB = True
except ImportError:
    try:
        import camb
        _CAMB_AVAILABLE = True
        _USING_MGCAMB = False
    except ImportError:
        _CAMB_AVAILABLE = False
        _USING_MGCAMB = False


class CAMBKinematics(ObservableEngineBase):
    """CAMB engine for ΛCDM and background-inert modified gravity models.

    For models where H(z) = ΛCDM but growth is modified (e.g. fQ_LCDM),
    use MGCAMBWrapper instead, which applies the correct growth-ratio
    correction on top of this engine's ΛCDM P(k,z).

    For models with genuine background modifications and scale-dependent
    gravitational slip (e.g. fQ_Hybrid), use MGCAMBKinematics.
    """

    capabilities = {"Pk_linear", "Pk_nonlinear", "sigma8_camb", "r_drag"}

    # Fixed cosmological parameters (not varied in current COSMIX runs).
    # Promote to ParameterManager entries if you want to vary them.
    _ns_default  = 0.9649    # Planck 2018 TT,TE,EE+lowE
    _tau_default = 0.0544    # Planck 2018
    _As_ref      = 2.1e-9   # Reference amplitude — only shape matters; norm via sigma80

    def __init__(self, model, theta, z_grid, k_min=1e-4, k_max=10.0, nk=200):
        if not _CAMB_AVAILABLE:
            raise ImportError(
                "[CAMBKinematics] Neither mgcamb nor camb is installed.\n"
                "  For modified gravity: pip install mgcamb\n"
                "  For ΛCDM only:        pip install camb"
            )

        self._model  = model
        self._theta  = theta
        self._z_grid = np.asarray(z_grid)
        self._nk     = nk
        self._k_min  = k_min
        self._k_max  = k_max

        # Extract COSMIX parameters
        pm = model.pm
        self._H0      = pm.get_value(theta, "H0")
        self._Omegam0 = pm.get_value(theta, "Omegam0")
        self._Omegab0 = pm.get_value(theta, "Omegab0")
        self._sigma80 = pm.get_value(theta, "sigma80")

        h     = self._H0 / 100.0
        ombh2 = self._Omegab0 * h**2                        # baryon physical density
        omch2 = (self._Omegam0 - self._Omegab0) * h**2     # CDM physical density

        # Run Boltzmann code once per parameter point; results cached in self._results
        self._results, self._sigma8_ref = self._run_camb(
            ombh2, omch2, self._H0, k_min, k_max, nk
        )
        # _z_internal: the z-grid actually passed to CAMB (ascending order)
        self._z_internal = np.unique(np.concatenate([self._z_grid, [0.0]]))

    def _run_camb(self, ombh2, omch2, H0, k_min, k_max, nk):
        """Set CAMB parameters, run, and return (CAMBdata, sigma8_ref).

        Returns (None, None) on CAMB failure so the pipeline can mark
        the theory cache invalid rather than raising.
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2,
            tau=self._tau_default
        )
        pars.InitPower.set_params(As=self._As_ref, ns=self._ns_default)

        z_camb = np.unique(np.concatenate([self._z_grid, [0.0]]))
        pars.set_matter_power(
            redshifts=sorted(z_camb, reverse=True),  # CAMB requires descending order
            kmax=k_max * 1.1,
            nonlinear=True
        )

        try:
            results = camb.get_results(pars)
        except Exception:
            return None, None

        sigma8_ref = results.get_sigma8_0()   # sigma8 at z=0 for the reference A_s
        return results, sigma8_ref

    def _rescale_factor(self):
        """(sigma80 / sigma8_ref)² — multiplicative rescaling for all P(k,z)."""
        if self._sigma8_ref is None or self._sigma8_ref <= 0:
            return None
        return (self._sigma80 / self._sigma8_ref) ** 2

    @property
    def sigma8_ref(self):
        """sigma8(z=0) predicted by CAMB at the fixed reference amplitude As_ref,
        for this instance's (H0, Omegam0, Omegab0) — i.e. BEFORE rescaling to the
        sampled sigma80.  This is the ℐ-encoding quantity used to compute the
        per-sample implied amplitude:

            A_s^implied = As_ref * (sigma80 / sigma8_ref)²

        Returns None if the CAMB run failed (see _run_camb)."""
        return self._sigma8_ref

    def Pk_linear(self, k, z):
        """Linear matter power spectrum P_lin(k,z).

        Parameters
        ----------
        k : array, h/Mpc
        z : array, redshifts

        Returns
        -------
        ndarray, shape (len(k), len(z)), units (Mpc/h)³
        """
        if self._results is None:
            return None
        scale = self._rescale_factor()
        if scale is None:
            return None

        kh, z_out, pk = self._results.get_matter_power_spectrum(
            minkh=max(k.min() * 0.9, 1e-5),
            maxkh=k.max() * 1.1,
            npoints=self._nk,
            var1="delta_tot",
            var2="delta_tot"
        )
        return self._interpolate_pk(kh, z_out, pk, k, z) * scale

    def Pk_nonlinear(self, k, z):
        """Non-linear matter power spectrum via halofit.

        Parameters
        ----------
        k : array, h/Mpc
        z : array, redshifts

        Returns
        -------
        ndarray, shape (len(k), len(z)), units (Mpc/h)³
        """
        if self._results is None:
            return None
        scale = self._rescale_factor()
        if scale is None:
            return None
        
        interp = self._results.get_matter_power_interpolator(
            nonlinear=True,
            var1="delta_tot",
            var2="delta_tot",
            hubble_units=True,      # P in (Mpc/h)^3
            k_hunit=True,           # k in Mpc/h
        )
        # P(z, kh, grid=True) returns shape (nz, nk)
        pk_grid = interp.P(
            np.asarray(z, dtype=float),
            np.asarray(k, dtype=float),
            grid=True
        )

        return pk_grid.T * scale        # transpose to (nk, nz) to match Pk_linear convention

    def _interpolate_pk(self, kh_camb, z_camb, pk_camb, k_target, z_target):
        """Bilinear log-log interpolation from CAMB native grid to (k_target, z_target).

        CAMB returns pk with shape (n_z, n_k) where z is in *descending* order.
        Output shape: (len(k_target), len(z_target)).
        """
        # Flip to ascending z
        z_asc  = z_camb[::-1]
        pk_asc = pk_camb[::-1, :]   # shape (nz, nk)

        log_kh = np.log(kh_camb)
        log_pk = np.log(np.clip(pk_asc, 1e-100, None))

        interp = RegularGridInterpolator(
            (z_asc, log_kh), log_pk,
            method="linear", bounds_error=False, fill_value=None
        )

        nk, nz = len(k_target), len(z_target)
        pts = np.array([[z_target[j], np.log(k_target[i])]
                        for i in range(nk) for j in range(nz)])
        return np.exp(interp(pts).reshape(nk, nz))

    def sigma8_camb(self, z):
        """sigma8(z) derived from CAMB P(k,z), rescaled by sigma80.

        Useful as a cross-check against GrowthKinematics.sigma8(z).
        Agreement to <1% for ΛCDM validates the pipeline.
        """
        if self._results is None:
            return None
        scale = self._rescale_factor()
        if scale is None:
            return None

        # get_sigma8() returns values at the z-grid passed to set_matter_power,
        # in the same (descending) order. Flip to ascending before interpolating.
        s8_vals  = self._results.get_sigma8()[::-1]    # now ascending in z
        z_sorted = np.sort(self._z_internal)            # ascending

        return np.interp(z, z_sorted, s8_vals) * np.sqrt(scale)

    def r_drag(self, z):
        """Sound horizon at drag epoch r_d, in Mpc/h.

        The z argument is accepted for API compatibility but is unused —
        r_d is a single scalar derived from the background evolution.
        """
        if self._results is None:
            return None
        rd = self._results.get_derived_params()["rdrag"]   # Mpc (not Mpc/h)
        h  = self._H0 / 100.0
        return np.full_like(np.asarray(z, dtype=float), rd * h)


# ══════════════════════════════════════════════════════════════════════════════
# MGCAMBKinematics — for models with genuine background + growth modifications
# ══════════════════════════════════════════════════════════════════════════════
class MGCAMBKinematics(CAMBKinematics):
    """MGCAMB engine for theories with scale-dependent gravitational slip.

    Extends CAMBKinematics by calling MGCAMB's set_MG_cosmo() to propagate
    a theory-specific μ(k,z) and Σ(k,z) into the Boltzmann hierarchy.
    Use this for models where H(z) ≠ ΛCDM or where the slip is k-dependent.

    For background-inert, k-independent models (fQ_LCDM), use the cheaper
    MGCAMBWrapper instead.

    Requires mgcamb to be installed:  pip install mgcamb

    The model must implement:
        mu_func(k, z)    — μ(k,z) = G_eff/G_N for the Poisson equation
        sigma_func(k,z)  — Σ(k,z) = lensing gravitational coupling
    These are passed as callables to MGCAMB's parameterisation system.
    """

    def __init__(self, model, theta, z_grid, k_min=1e-4, k_max=10.0, nk=200):
        if not _USING_MGCAMB:
            raise ImportError(
                "[MGCAMBKinematics] requires mgcamb: pip install mgcamb"
            )
        # Store model reference before calling super().__init__ which calls _run_camb
        self._mg_model = model
        self._mg_theta = theta
        super().__init__(model, theta, z_grid, k_min, k_max, nk)

    def _run_camb(self, ombh2, omch2, H0, k_min, k_max, nk):
        """Override to inject MG parameters via set_mgparams() before get_results().

        The model must implement mg_camb_params(theta, bg_engine) returning a dict
        of keyword arguments accepted by CAMBparams.set_mgparams(). These map to
        MGCAMB's built-in parametric models (mu/Sigma, BZ, Linder, f(R), spline).
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2,
            tau=self._tau_default
        )
        pars.InitPower.set_params(As=self._As_ref, ns=self._ns_default)

        z_camb = np.unique(np.concatenate([self._z_grid, [0.0]]))
        pars.set_matter_power(
            redshifts=sorted(z_camb, reverse=True),
            kmax=k_max * 1.1,
            nonlinear=True
        )

        # Inject MG parameters using MGCAMB's actual set_mgparams() API.
        # The model provides the parameter dict; we do not assume any structure.
        if hasattr(self._mg_model, "mg_camb_params"):
            # Build a temporary background engine for models that need H(z) to
            # compute their MG parameters (e.g. spline μ(z) nodes)
            from cosmix.core.BackgroundKinematics import BackgroundKinematics
            from cosmix.core.BackgroundConfiguration import BackgroundConfig
            from dataclasses import replace as dc_replace
            bg_config = dc_replace(
                self._mg_model.background_config(),
                z_max_extended=max(z_camb.max(), 3.0)
            )
            bg_tmp = BackgroundKinematics(
                model=self._mg_model,
                theta=self._mg_theta,
                config=bg_config
            )
            mg_kwargs = self._mg_model.mg_camb_params(self._mg_theta, bg_tmp)
            try:
                pars.set_mgparams(**mg_kwargs)
            except Exception:
                return None, None

        try:
            results = camb.get_results(pars)
        except Exception:
            return None, None

        return results, results.get_sigma8_0()


# ══════════════════════════════════════════════════════════════════════════════
# MGCAMBWrapper — growth-ratio correction for background-inert models
# ══════════════════════════════════════════════════════════════════════════════
class MGCAMBWrapper(ObservableEngineBase):
    """Applies a growth-ratio modified gravity correction to CAMB P(k,z).

    Exact for background-inert, scale-independent models (fQ_LCDM family):
        P_fQ(k,z) = P_ΛCDM(k,z) × [D_fQ(z) / D_ΛCDM(z)]²

    Since fQ_LCDM has H(z) = ΛCDM, the transfer function T(k) is identical,
    and the only modification is in the growth factor D(z) via μ_G(z).
    GrowthKinematics already solves the modified growth ODE — delta(z) is
    D_fQ(z)/D_fQ(0), which equals D_fQ(z)/D_ΛCDM(0) ≈ D_fQ(z)/D_ΛCDM(0)
    when the two are normalised identically at z=0 by sigma80.

    For k-dependent or background-modifying theories, use MGCAMBKinematics.
    """

    capabilities = {"Pk_linear", "Pk_nonlinear", "sigma8_camb", "r_drag"}

    def __init__(self, camb_engine, growth_engine):
        """
        Parameters
        ----------
        camb_engine   : CAMBKinematics — pre-computed ΛCDM P(k,z)
        growth_engine : GrowthKinematics — modified delta(z) from the fQ ODE
        """
        self._camb   = camb_engine
        self._growth = growth_engine

    def Pk_linear(self, k, z):
        """P_fQ_lin(k,z) = P_ΛCDM_lin(k,z) × [D_fQ(z)/D_fQ(0)]²."""
        pk_z0 = self._camb.Pk_linear(k, np.zeros(1))   # shape (nk, 1) — no z-evolution
        if pk_z0 is None:
            return None
        ratio = self._growth.delta(z)                   # D_fQ(z)/D_fQ(0), =1 at z=0
        
        return pk_z0 * ratio[np.newaxis, :] ** 2        # (nk, nz) 

    def Pk_nonlinear(self, k, z):
        """Approximate non-linear P_fQ(k,z) using halofit ΛCDM shape × growth ratio.

        This is an approximation: halofit was calibrated for GR and the
        non-linear growth in fQ_LCDM will differ at the ~few-percent level.
        Sufficient for weak lensing forecasts; use a full MG halofit when
        precision constraints are required.
        """
        pk_nl_z0 = self._camb.Pk_nonlinear(k, np.zeros(1))
        if pk_nl_z0 is None:
            return None
        ratio = self._growth.delta(z)

        return pk_nl_z0 * ratio[np.newaxis, :] ** 2

    def sigma8_camb(self, z):
        return self._camb.sigma8_camb(z)

    def r_drag(self, z):
        return self._camb.r_drag(z)

    @property
    def sigma8_ref(self):
        return self._camb.sigma8_ref

    
