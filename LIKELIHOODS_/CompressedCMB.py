"""CompressedCMB — Planck 2018 compressed CMB likelihood.

Implements the four-parameter compressed likelihood introduced in
Chen, Huang & Wang (2019) [arXiv:1808.05724].  The compressed data
vector is:

    X = [R, l_a, omega_b, n_s]

where:
    R     = sqrt(Omega_m0) * (H0/c) * chi(z_star)
            Shift parameter encoding the angular size of the sound horizon.
    l_a   = pi * chi(z_star) / r_s(z_star)
            Acoustic scale parameter.
    omega_b = Omega_b * h^2
            Physical baryon density.
    n_s   = scalar spectral index.

Physics notes
-------------
Decoupling redshift z_star is computed analytically via the
Hu & Sugiyama (1996) fitting formula (accurate to ~0.3%).

chi(z_star) is evaluated by integrating the model's H(z) from z=0 to
z_star using the BackgroundKinematics engine via the theory cache.
This is fully model-dependent and handles IDE/MG H(z) correctly.

r_s(z_star) is computed analytically using the radiation+matter
approximation for H(z).  This is justified because at z > z_star ~ 1090
all dark energy / IDE effects are completely negligible (~10^{-4} level).

Data
----
Expected .npz file keys:
    mean_vector : shape (4,)  -- [R, l_a, omega_b, n_s] Planck 2018 bestfit
    inv_cov     : shape (4,4) -- inverse covariance matrix

Default file: DATA_/CMB_compressed/planck_2018.npz

References
----------
- Chen, Huang & Wang (2019), JCAP 02 (2019) 028; arXiv:1808.05724
- Hu & Sugiyama (1996), ApJ 471:542; arXiv:astro-ph/9510117
- Planck 2018 Results VI, A&A 641 A6 (2020); arXiv:1807.06209
"""

import numpy as np
import pandas as pd
from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase
from CORE_.ParameterManager_ import Parameter, GaussianPrior
from Constants import c, Omegar0

_DATA_CompCMB = Path(__file__).resolve().parent.parent / "DATA_" / "Compressed_CMB" 
Default_data_file = _DATA_CompCMB / "compressed_cmb_mean.txt"
Default_cov_file = _DATA_CompCMB / "compressed_cmb_cov.txt"

# Fixed z-grid requested from the background engine.
# Covers 0 → 1100 with geometric spacing to resolve the low-z dark energy
# era and the radiation era near decoupling equally well.
# This also triggers z_max_extended in build_engines (see CosmologyModelBase).
_CMB_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 1100.0, 499)])


class CompressedCMB(LikelihoodBase):
    """Planck 2018 compressed CMB likelihood over [R, l_a, omega_b, n_s]."""

    name = "CompCMB"

    def __init__(self, pm, data_file=None, cov_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file
        if cov_file is None:
            cov_file = Default_cov_file

        data = pd.read_csv(data_file, sep=r"\s+", header=None, names=["value", "label"]) 
        self.d_vec = data["value"]   # shape (4,)
        self.data_size = len(self.d_vec)
        cov = np.loadtxt(cov_file).reshape(self.data_size, self.data_size)
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        
        self.produce_residuals = True

    @classmethod
    def declare_parameters(cls):
        """Declare omega_b and n_s as likelihood-owned nuisance parameters."""
        return [
            Parameter(
                name="omega_b",
                latex=r"\omega_b",
                prior=GaussianPrior(mean=0.02237, sig=0.00015, low=0.01, high=0.05),
                role="nuisance",
                status="free",
                proposed_scale=0.0001
            ),
            Parameter(
                name="n_s",
                latex=r"n_s",
                prior=GaussianPrior(mean=0.9649, sig=0.0042, low= 0.5, high=1.5),
                role="nuisance",
                status="free",
                proposed_scale=0.003
            ),
        ]

    def get_requirements(self):
        """
        Request H(z) on a grid spanning 0 to z_star.

        The high-z extent (z ~ 1100) triggers z_max_extended in
        CosmologyModelBase.build_engines so that BackgroundKinematics
        correctly covers the decoupling epoch.
        """
        return {"H": _CMB_Z_GRID}

    def lnlike(self, theta, theory):
        """
        Evaluate the compressed CMB log-likelihood.

        Steps
        -----
        1. Compute z_star via the Hu & Sugiyama (1996) fitting formula.
        2. Integrate H(z) from the theory cache to obtain chi(z_star).
        3. Compute r_s(z_star) analytically (radiation+matter dominated
           at z > z_star; model-independent to 1 part in 10^4).
        4. Build [R, l_a, omega_b, n_s] and evaluate chi^2.
        """
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        H0 = self.pm.get_value(theta, "H0") 
        omega_b = self.pm.get_value(theta, "omega_b")
        n_s     = self.pm.get_value(theta, "n_s")
        h = H0 / 100.0
        omega_m = Omegam0 * h**2

        # ------------------------------------------------------------------
        # 1. Decoupling redshift (Hu & Sugiyama 1996, Eq. 17)
        # ------------------------------------------------------------------
        g1 = 0.0783 * omega_b**(-0.238) / (1.0 + 39.5 * omega_b**0.763)
        g2 = 0.560  / (1.0 + 21.1 * omega_b**1.81)
        z_star = 1048.0 * (1.0 + 0.00124 * omega_b**(-0.738)) * (1.0 + g1 * omega_m**g2)

        # Sanity check: z_star should be near 1090; bail if wildly off
        if not (800.0 < z_star < 1200.0):
            return -np.inf

        # ------------------------------------------------------------------
        # 2. Comoving distance chi(z_star) from the model's H(z)
        # ------------------------------------------------------------------
        # Build a fine integration grid from 0 to z_star.
        # np.interp inside theory.eval handles z=0 correctly (returns H0).
        z_chi = np.concatenate([[0.0], np.geomspace(1e-4, z_star, 399)])
        H_chi = theory.eval("H", z_chi)            # [km/s/Mpc]
        chi_star = np.trapz(c / H_chi, z_chi)     # [Mpc]

        if not np.isfinite(chi_star) or chi_star <= 0.0:
            return -np.inf

        # ------------------------------------------------------------------
        # 3. Sound horizon r_s(z_star)
        # ------------------------------------------------------------------
        # At z > z_star dark energy / IDE contributions are < 10^{-4};
        # use the radiation+matter analytical H(z) here.
        #
        # Omegar_rs: use the Eisenstein & Hu z_eq formula (Zhai & Wang 2018 Eq.28)
        #   z_eq = 2.5e4 * omega_m * (T/2.7)^{-4},  Omega_r = Omega_m / (1+z_eq)
        # This is the formula the Planck 2018 chain itself uses to compute la.
        # Using the fixed Constants.Omegar0 instead creates a ~5-unit la bias.
        #
        # z_max: must go to at least 1e7.  The tail from z_max to infinity
        # contributes ~c/(sqrt(3)*H0*sqrt(Or)*z_max) Mpc; at z_max=1e5 this
        # is ~2.4 Mpc (a 1.7% error in rs), while at z_max=1e7 it is ~0.03 Mpc.
        T_cmb = 2.7255
        z_eq  = 2.5e4 * omega_m * (T_cmb / 2.7)**(-4)
        Omegar_rs = Omegam0 / (1.0 + z_eq)
        z_rs  = np.geomspace(z_star, 1e7, 600)
        R_b   = 31500.0 * omega_b * (T_cmb / 2.7)**(-4) / (1.0 + z_rs)
        c_s   = c / np.sqrt(3.0 * (1.0 + R_b))
        E_rs  = np.sqrt(Omegam0 * (1.0 + z_rs)**3 + Omegar_rs * (1.0 + z_rs)**4)
        H_rs  = H0 * E_rs
        rs_star = np.trapz(c_s / H_rs, z_rs)

        if not np.isfinite(rs_star) or rs_star <= 0.0:
            return -np.inf

        # ------------------------------------------------------------------
        # 4. Shift parameters and chi^2
        # ------------------------------------------------------------------
        R_model  = np.sqrt(Omegam0) * (H0 / c) * chi_star
        la_model = np.pi * chi_star / rs_star

        th_vector = np.array([R_model, la_model, omega_b, n_s])
        delta = th_vector - self.d_vec
        chi2  = delta @ self.inv_cov @ delta

        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2

    def norm_term(self):
        return 0.0

    def get_theory_components(self, theta, theory, z_override=None):
        """Return scalar shift parameter comparisons for diagnostics."""
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        H0 = self.pm.get_value(theta, "H0")
        omega_b = self.pm.get_value(theta, "omega_b")
        n_s     = self.pm.get_value(theta, "n_s")
        h = H0 / 100.0
        omega_m = Omegam0 * h**2

        g1 = 0.0783 * omega_b**(-0.238) / (1.0 + 39.5 * omega_b**0.763)
        g2 = 0.560  / (1.0 + 21.1 * omega_b**1.81)
        z_star = 1048.0 * (1.0 + 0.00124 * omega_b**(-0.738)) * (1.0 + g1 * omega_m**g2)

        z_chi   = np.concatenate([[0.0], np.geomspace(1e-4, z_star, 399)])
        H_chi   = theory.eval("H", z_chi)
        chi_star = np.trapz(c / H_chi, z_chi)

        T_cmb = 2.7255
        z_eq  = 2.5e4 * omega_m * (T_cmb / 2.7)**(-4)
        Omegar_rs = Omegam0 / (1.0 + z_eq)
        z_rs  = np.geomspace(z_star, 1e7, 600)
        R_b   = 31500.0 * omega_b * (T_cmb / 2.7)**(-4) / (1.0 + z_rs)
        c_s   = c / np.sqrt(3.0 * (1.0 + R_b))
        E_rs  = np.sqrt(Omegam0 * (1.0 + z_rs)**3 + Omegar_rs * (1.0 + z_rs)**4)
        H_rs  = H0 * E_rs
        rs_star = np.trapz(c_s / H_rs, z_rs)

        R_model  = np.sqrt(Omegam0) * (H0 / c) * chi_star
        la_model = np.pi * chi_star / rs_star

        th_vec   = np.array([R_model, la_model, omega_b, n_s])
        labels   = ["R", r"$l_a$", r"$\omega_b$", r"$n_s$"]
        sigma    = np.sqrt(np.diag(np.linalg.inv(self.inv_cov)))

        # Return numeric indices as x so the residual plot doesn't create a
        # categorical axis that would corrupt shared-axis panels.
        # _param_labels is read by Visualization.residual() for tick labeling.
        self._param_labels = labels
        return {"CMB": (np.arange(len(labels), dtype=float), self.d_vec, th_vec, sigma)}
