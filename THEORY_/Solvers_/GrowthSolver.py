"""GrowthSolver — fast RK4 integrator for the linear growth ODE.

Solves

    δ'' + (2 + d ln H / d ln a) δ' − 1.5 μ_G(z) Ω_m(z) δ = 0

forward from N_ini = ln a_ini = −7 (z ≈ 1096) to N = 0 (z = 0) on a
uniform grid of 250 steps.  The RK4 loop is JIT-compiled with Numba.

After integration the solution is normalised so δ(z=0) = 1.  The raw
(pre-normalisation) amplitude D(0) is stored as ``solver.unnorm_D0``
for the implied-As diagnostic.
"""

import numpy as np
from numba import njit

@njit(fastmath=True)
def _fast_rk4_loop(N_steps, dN, N_ini, friction_array, source_array):
    delta = np.zeros(N_steps)
    ddelta = np.zeros(N_steps)
    # initial condition (delta ~ a during matter domination)
    delta[0] = np.exp(N_ini)
    ddelta[0] = np.exp(N_ini)

    for i in range(N_steps - 1):
        y0 = delta[i]
        v0 = ddelta[i]

        f_i = friction_array[i]
        f_ip1 = friction_array[i+1]
        s_i = source_array[i]
        s_ip1 = source_array[i+1]

        # midpoint approximation
        f_mid = 0.5 * ( f_i + f_ip1 )
        s_mid = 0.5 * ( s_i + s_ip1 )

        # k1
        k1_y = v0
        k1_v = -f_i * v0 + s_i * y0
        # k2
        k2_y = v0 + 0.5 * dN * k1_v
        k2_v = -f_mid * (v0 + 0.5 * dN * k1_v) + s_mid * (y0 + 0.5 * dN * k1_y)
        # k3
        k3_y = v0 + 0.5 * dN * k2_v
        k3_v = -f_mid * (v0 + 0.5 * dN * k2_v) + s_mid * (y0 + 0.5 * dN * k2_y)
        # k4
        k4_y = v0 + dN * k3_v
        k4_v = -f_ip1 * (v0 + dN * k3_v) + s_ip1 * (y0 + dN * k3_y)

        delta[i+1] = y0 + (dN / 6.0) * ( k1_y + 2*k2_y + 2*k3_y + k4_y )
        ddelta[i+1] = v0 + (dN / 6.0) * ( k1_v + 2*k2_v + 2*k3_v + k4_v )

    return delta, ddelta



class GrowthSolver:
    """
    Highly optimized linear growth solver using pre-computed background arrays
    and a custom vectorized RK4 integrator to bypass Python function overhead.
    """

    def __init__(self, background, model, theta, ode_solver=None):
        self.bg = background
        self.model = model
        self.theta = theta
        #self.ode = ode_solver


    def solve(self):
        z_grid = self.bg.z_grid()
        N_grid = -np.log(1.0 + z_grid)        # N = ln a = -ln (1 + z)


        # 1. Define a tight, fixed integration grid. 250 points suffices for the smooth linear growth.
        N_ini = -7.0
        N_end = 0.0
        N_steps = 250
        # integrating forward in time, i.e. from N_ini to N_end
        N_eval = np.linspace(N_ini, N_end, N_steps)
        dN = N_eval[1] - N_eval[0]
        z_eval = np.exp(-N_eval) - 1.0


        # 2. Pre-compute physics arrays once eliminating object lookup and interpolation overhead inside the ODE loop.
        dlnH_dN_arr = self.bg.dlnH_dN(z_eval)
        Omegam_arr = self.bg.Omegamz(z_eval)
        mu_arr = self.bg.muG(z_eval)
        # growth equation: delta'' + friction * delta' - source * delta = 0
        friction_arr = 2.0 + dlnH_dN_arr
        source_arr = 1.5 * Omegam_arr * mu_arr


        # 3. Fast custom RK4 integration loop
        delta, ddelta = _fast_rk4_loop(
            N_steps=N_steps,
            dN=dN,
            N_ini=N_ini,
            friction_array=friction_arr,
            source_array=source_arr
        )

        # 4. safety check & normalization
        norm = delta[-1]
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("[GrowthSolver] Growth integration diverged --- invalid parameter space!")

        # Expose the raw (unnormalized) growth amplitude D(z=0) before dividing it out.
        # Diagnostic use: As_implied ∝ (sigma80_fit / sigma80_ref)² × (D_ref / D_model)²
        self.unnorm_D0 = norm

        delta /= norm
        ddelta /= norm


        # 5. Interpolate back to the master background grid
        # N_eval is sorted ascending and thus, np.interp works perfectly.
        self.delta_grid = np.interp(N_grid, N_eval, delta)
        self.ddelta_grid = np.interp(N_grid, N_eval, ddelta)
        self.z_grid = z_grid


    def delta(self, z):
        return np.interp(z, self.z_grid, self.delta_grid)
    
    def f(self, z):                 # f delta = d delta / d N
        delta = self.delta(z)
        ddelta = np.interp(z, self.z_grid, self.ddelta_grid)

        return ddelta / delta