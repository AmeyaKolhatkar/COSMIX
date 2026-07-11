import numpy as np
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod
from cosmix.core.CosmologyModelBase import CosmologyModelBase
from cosmix.core.BackgroundConfiguration import BackgroundConfig
from cosmix.core.BackgroundKinematics import BackgroundKinematics
from cosmix.core.ParameterManager_ import Parameter, Prior, UniformPrior, GaussianPrior
from cosmix.Constants import c, Omegar0


class _ScipyODEProblem:
    def __init__(self, rhs, y0, extract):
        self.rhs = rhs                  #callable(z, y) -> dy/dz     (for solve_ivp)
        self.y0 = y0                    # [Q0, P0, gamma0_val]       (initial state at z=0)
        self.extract = extract          #callable(sol, z_grid) -> dict

    def solve(self, z_grid):
        t_span = (float(z_grid[0]), float(z_grid[-1]))
        sol = solve_ivp(
            self.rhs,
            t_span,
            self.y0,
            method='RK45',
            t_eval=z_grid,
            rtol=1e-8,
            atol=1e-10,
            dense_output=False
        )
        if not sol.success:
            raise RuntimeError(
                f"_ScipyODEProblem: integration failed — {sol.message}"
            )
        return self.extract(sol, z_grid)



class CurvedfQBase(CosmologyModelBase):
    name = "curvedfQBase"

    def __init__(self, pm):
        super().__init__(pm)

    # ── Abstract interface (subclasses must implement) ──────────────────────
    @abstractmethod
    def f(self, Q) -> float: ...
    @abstractmethod
    def f_prime(self, Q) -> float: ...
    @abstractmethod
    def f_double_prime(self, Q) -> float: ...
    @abstractmethod
    def f_triple_prime(self, Q) -> float: ...

    # ── Step 3: Friedmann inversion ──────────────────────────────────────────
    def _solve_H(self, z, Q, P, gamma, H0, Omegam0, Omegak):
        """Solve the modified Friedmann eq. (78a) for H at given (z, Q, P, γ).

        With N=1 and Q̇ = -H(1+z)P substituted, eq. 78a becomes:
            A·H² + B·H + C = 0
        where
            A = 3 f'(Q)
            B = (3/2) P f''(Q) (1+z) [γ(1+z)² + k/γ]
            C = (1/2)(f - Q f') + 3k(1+z)² f' - ρ(z)

        Returns the positive physical root.
        """
        k_phys = -Omegak * H0**2
        zp1    = 1.0 + z
        rho    = 3.0 * H0**2 * (Omegam0 * zp1**3 + Omegar0 * zp1**4)

        f_val = self.f(Q)
        fp    = self.f_prime(Q)
        fpp   = self.f_double_prime(Q)

        psi = gamma * zp1**2 + k_phys / gamma   # γ(1+z)² + k/γ

        A = 3.0 * fp
        B = 1.5 * P * fpp * zp1 * psi
        C = 0.5 * (f_val - Q * fp) + 3.0 * k_phys * zp1**2 * fp - rho

        if A == 0.0:
            # f'=0: degenerate quadratic → linear equation
            if B == 0.0:
                raise RuntimeError(
                    f"Friedmann eq. degenerate at z={z:.4f}: A=B=0."
                )
            return -C / B

        disc = B**2 - 4.0 * A * C
        if disc < 0.0:
            raise RuntimeError(
                f"Friedmann eq. has no real solution at z={z:.4f}: disc={disc:.4e}."
            )
        H_p = (-B + np.sqrt(disc)) / (2.0 * A)
        H_m = (-B - np.sqrt(disc)) / (2.0 * A)
        if H_p > 0.0:
            return H_p
        if H_m > 0.0:
            return H_m
        raise RuntimeError(
            f"No positive H root at z={z:.4f}: H+={H_p:.4e}, H-={H_m:.4e}."
        )

    # ── Step 4: ODE RHS ──────────────────────────────────────────────────────
    def _ode_rhs(self, z, y, H0, Omegam0, Omegak):
        """RHS of the ODE system for state y = [Q, P=dQ/dz, γ].

        Returns dy/dz = [P, P', g] where:

          dQ/dz = P                (definition)
          dγ/dz = g                from Dimakis et al. (2205.04680) eq. 77
                                   inverted for γ̇, then converted to z-space
          dP/dz = P'               from eq. 79 + differentiated Friedmann,
                                   solved as a 2×2 linear system in (P', H')

        det(M) = 6 f'(Q) f''(Q) H³ (1+z)² α₁,  where α₁ = 1 + k a²/γ².
        The system is singular when f'=0, f''=0, H=0, or α₁=0.
        For models with f''=0 everywhere (e.g. linear f), do not use this
        ODE — override background_problem in the subclass instead.
        """
        Q, P, gamma = y

        k_phys  = -Omegak * H0**2
        zp1     = 1.0 + z
        zp1_sq  = zp1 * zp1

        # ── Model functions ──────────────────────────────────────────────────
        f_val = self.f(Q)
        fp    = self.f_prime(Q)
        fpp   = self.f_double_prime(Q)
        fppp  = self.f_triple_prime(Q)

        # Guard: f′(Q) must remain positive throughout the integration.
        # If it crosses zero the system enters a ghost sector; det(M) → 0
        # and RK45 would subdivide indefinitely rather than fail cleanly.
        # Raising here propagates through solve_ivp → compute_theory (caught)
        # and returns -inf to the sampler — fast rejection instead of a hang.
        if fp <= 1e-4:
            raise RuntimeError(
                f"f'(Q) ≤ 1e-4 at z={z:.4f}: f'={fp:.4e}. "
                f"Ghost sector crossing — rejecting trajectory."
            )

        # ── H from Friedmann (Step 3) ────────────────────────────────────────
        H = self._solve_H(z, Q, P, gamma, H0, Omegam0, Omegak)

        # ── Auxiliary scalars ────────────────────────────────────────────────
        gamma_sq = gamma * gamma
        beta     = k_phys / (gamma_sq * zp1_sq)   # k a²/γ²
        alpha1   = 1.0 + beta                       # 1 + k a²/γ²
        psi      = gamma * zp1_sq + k_phys / gamma  # γ(1+z)² + k/γ

        # Guard: α₁ = 1 + ka²/γ² must be positive; α₁ ≤ 0 means the
        # non-trivial curved connection has become degenerate (geometry
        # singularity).  Same early-rejection logic as the fp guard above.
        if alpha1 <= 1e-4:
            raise RuntimeError(
                f"α₁ = 1 + ka²/γ² ≤ 1e-4 at z={z:.4f}: α₁={alpha1:.4e}. "
                f"Connection singularity — rejecting trajectory."
            )

        # ── dγ/dz from eq. 77 inverted ──────────────────────────────────────
        # γ̇ · 3[(1+z)² + k/γ²] = Q + 6H² - 3Hγ(1+z)² - 6k(1+z)² + 9kH/γ
        # g = dγ/dz = γ̇ / (dz/dt),  dz/dt = −H(1+z)
        num_g = (Q + 6.0*H*H
                 - 3.0*H*gamma*zp1_sq
                 - 6.0*k_phys*zp1_sq
                 + 9.0*k_phys*H/gamma)
        den_g = -3.0 * H * zp1 * (zp1_sq + k_phys / gamma_sq)
        g = num_g / den_g

        # ── b₁: RHS of eq. 79 (P' and H' terms removed) ────────────────────
        # [eq. 79] → b₁ = −fppp H²(1+z)²P²α₁ − 2 fpp H²(1+z)P [−β + (1+z)g/γ]
        b1 = (-fppp * H*H * zp1_sq * P*P * alpha1
              - 2.0 * fpp * H*H * zp1 * P * (-beta + zp1 * g / gamma))

        # ── b₂: RHS of differentiated Friedmann (P' and H' terms removed) ──
        # ψ' = dψ/dz
        psi_prime = g * zp1_sq + 2.0*gamma*zp1 - k_phys*g/gamma_sq
        # ρ' = dρ/dz
        rho_prime = 3.0*H0*H0*(3.0*Omegam0*zp1_sq + 4.0*Omegar0*zp1_sq*zp1)
        # A' = 3 fpp P,  B'_{no P'} = (3P/2)[fppp P (1+z)ψ + fpp(ψ + (1+z)ψ')]
        A_prime       = 3.0 * fpp * P
        B_prime_no_Pp = 1.5 * P * (fppp*P*zp1*psi + fpp*(psi + zp1*psi_prime))
        # C' = −Q fpp P/2 + 6k(1+z) f' + 3k(1+z)² fpp P − ρ'
        C_prime = (-0.5*Q*fpp*P
                   + 6.0*k_phys*zp1*fp
                   + 3.0*k_phys*zp1_sq*fpp*P
                   - rho_prime)
        b2 = -(A_prime*H*H + H*B_prime_no_Pp + C_prime)

        # ── 2×2 linear system for (P', H') ──────────────────────────────────
        # [ m11  m12 ] [P']   [b1]
        # [ m21  m22 ] [H'] = [b2]
        #
        # det(M) = 6 f' f'' H³ (1+z)² α₁
        det_M = 6.0 * fp * fpp * H*H*H * zp1_sq * alpha1
        if det_M == 0.0:
            raise RuntimeError(
                f"ODE matrix singular at z={z:.4f} "
                f"(f'={fp:.4e}, f''={fpp:.4e}, α₁={alpha1:.4e})."
            )
        B_frie = 1.5 * P * fpp * zp1 * psi        # = B from _solve_H
        m12    = fpp * H * zp1_sq * alpha1 * P
        m21    = 1.5 * H * fpp * zp1 * psi
        m22    = 6.0 * fp * H + B_frie

        P_prime = (m22 * b1 - m12 * b2) / det_M

        return np.array([P, P_prime, g], dtype=np.float64)

    def _initial_conditions(self, H0, Omegam0, Omegak, gamma_0):
        # Curvature parameter in the same units as H0² (a₀=1 convention).
        # Sign: k = -Ωk·H₀², so open universe (Ωk>0) → k<0.
        k_phys = -Omegak * H0**2

        # Q₀ from Dimakis et al. (2205.04680) eq. 77 at z=0
        # with N=1, a=1, and quasi-static IC: γ̇₀ = 0.
        # Q = -6H² + 3Hγ/a² + (3/a² + 3k/γ²)γ̇ + 6k/a² - 9kH/γ
        # Setting γ̇₀=0 and a₀=1:
        if gamma_0 == 0.0:
            raise ValueError("gamma_0 must be non-zero for the curved f(Q) connection.")
        Q0 = (-6.0 * H0**2
              + 3.0 * H0 * gamma_0
              + k_phys * (6.0 - 9.0 * H0 / gamma_0))

        P0 = 0.0   # dQ/dz|₀ — quasi-static IC consistent with γ̇₀=0

        return np.array([Q0, P0, gamma_0], dtype=np.float64)

    def _set_params_from_theta(self, theta):
        """Hook for subclasses to cache model-specific parameters before any
        f(Q) evaluation.  Called automatically at the top of both
        check_physicality and background_problem.  Default is a no-op;
        override in concrete subclasses that have extra free parameters."""

    def background_problem(self, theta, z_grid):
        self._set_params_from_theta(theta)
        H0      = self.pm.get_value(theta, "H0")
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        Omegak  = self.pm.get_value(theta, "Omegak")
        gamma_0 = self.pm.get_value(theta, "gamma_0")

        y0  = self._initial_conditions(H0, Omegam0, Omegak, gamma_0)
        rhs = lambda z, y: self._ode_rhs(z, y, H0, Omegam0, Omegak)

        def extract(sol, z_grid):
            Q_arr     = sol.y[0]
            P_arr     = sol.y[1]
            gamma_arr = sol.y[2]
            H_arr = np.array([
                self._solve_H(zv, Q, P, gam, H0, Omegam0, Omegak)
                for zv, Q, P, gam in zip(z_grid, Q_arr, P_arr, gamma_arr)
            ])
            return {"H": H_arr, "Q": Q_arr, "gamma": gamma_arr}

        return _ScipyODEProblem(rhs=rhs, y0=y0, extract=extract)

    def background_config(self):
        # 60-point grid is sufficient for CC/BAO/SN datasets up to z=3.
        # nz=150 provided no accuracy benefit over 60 for smooth H(z) curves
        # and tripled the per-step ODE evaluation count.
        return BackgroundConfig(z_max=3.0, nz=60, integration_method="trapz")

    def muG(self, z, theta, bg_engine):
        raise NotImplementedError(
            "muG is not defined for the curved f(Q) connection. "
            "The scalar perturbation sector carries a ghost for generic "
            "connection function C (arXiv:2311.05495 §6.3 eq. 276; "
            "arXiv:2311.04201). No quasi-static effective Newton's constant "
            "can be derived from the current perturbation theory."
        )

    def check_physicality(self, theta):
        self._set_params_from_theta(theta)
        gamma_0 = self.pm.get_value(theta, "gamma_0")
        if gamma_0 == 0.0:
            return False

        H0     = self.pm.get_value(theta, "H0")
        Omegak = self.pm.get_value(theta, "Omegak")

        # α₁(z=0) = 1 + k/γ₀² must be positive at the starting point.
        # If it isn't, the curved connection is immediately singular and the
        # ODE cannot be integrated at all.
        k_phys = -Omegak * H0**2
        alpha1_z0 = 1.0 + k_phys / (gamma_0 * gamma_0)
        if alpha1_z0 <= 0.0:
            return False

        Omegam0 = self.pm.get_value(theta, "Omegam0")

        # f′(Q₀) > 0 ensures positive effective gravitational coupling at z=0.
        try:
            y0 = self._initial_conditions(H0, Omegam0, Omegak, gamma_0)
            if self.f_prime(y0[0]) <= 1e-4:
                return False
        except (ValueError, RuntimeError):
            return False

        return True

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="H0",
                latex=r"H_0",
                prior=UniformPrior(low=50.0, high=90.0),
                role="cosmo",
                status="free",
                proposed_scale=1.0
            ),
            Parameter(
                name="Omegam0",
                latex=r'\Omega_{m0}',
                prior=UniformPrior(low=0.0, high=1.0),
                role="cosmo",
                status="free",
                proposed_scale=0.015
            ),
            Parameter(
                name="Omegak",
                latex=r'\Omega_{k0}',
                prior=UniformPrior(low=-1.0, high=1.0),
                role="cosmo",
                status="fixed",
                value=0.0,
                proposed_scale=0.01
            ),
            Parameter(
                name="gamma_0",
                latex=r'\gamma_0',
                prior=UniformPrior(low=-1.0, high=1.0),
                role="cosmo",
                status="free",
                proposed_scale=0.05
            )
        ]
