"""
!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!

Core Limber Engine
"""

def limber_Cl(ell_arr, k_grid, z_grid, Pk_2d, chi_z, q_i, q_j, Sigma_z=None):
    """
    Compute C_ell^ij via the Limber approximation.

    Parameters
    ----------
    ell_arr  : (n_ell,) angular multipoles
    k_grid   : (nk,) wavenumbers h/Mpc
    z_grid   : (nz,) redshifts
    Pk_2d    : (nk, nz) P_fQ(k,z) from MGCAMBWrapper
    chi_z    : callable chi(z) [Mpc/h] from BackgroundKinematics
    q_i, q_j : (nchi,) lensing efficiency kernels on chi grid
    Sigma_z  : (nchi,) lensing slip Σ(z), default ones

    Returns
    -------
    Cl : (n_ell,) array
    """
    # Build z(chi) inverse mapping via interpolation
    # For each ell: k = (ell+0.5)/chi, evaluate P(k, z(chi)) on chi grid
    # Integrate with Simpson's rule