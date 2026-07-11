# Custom RK4 solver

import numpy as np
from numba import njit

@njit
def numba_rk4(rhs, z_grid, y0, params):
    """
    Bare-metal generic RK4 integrator.
    Takes a jitted RHS function and an array of parameters
    """
    n_pts = len(z_grid)
    n_vars = len(y0)

    # Output array; shape : ((nvars, npts))
    y_out = np.zeros((n_vars, n_pts), dtype=np.float64)
    y_out[:, 0] = y0

    # current state
    y_curr = np.copy(y0)

    for i in range(n_pts - 1):
        x = z_grid[i]
        dx = z_grid[i+1] - x

        # evaluate the models physical callback
        k1 = dx * np.asarray(rhs(x, y_curr, params))
        k2 = dx * np.asarray(rhs(x+0.5*dx, y_curr+0.5*k1, params))
        k3 = dx * np.asarray(rhs(x+0.5*dx, y_curr+0.5*k2, params))
        k4 = dx * np.asarray(rhs(x+dx, y_curr+k3, params))

        y_curr = y_curr + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        y_out[:, i+1] = y_curr

    return y_out