"""BackgroundConfiguration — immutable numerical configuration for background solves.

A frozen dataclass passed to BackgroundKinematics.  Externalising these
numerical choices ensures background solves are deterministic, hashable,
and cacheable.  Use ``dataclasses.replace`` to derive modified copies:

    from dataclasses import replace
    hi_z_config = replace(base_config, z_max_extended=1100.0)
"""

from dataclasses import dataclass

@dataclass(frozen=True)                     # frozen = True : guarantees immutability
class BackgroundConfig:
    """
    Numerical configuration for background cosmology. Immutable and Hashable.
    This class exists to:
        - externalize all numerical choices
        - make background solves deterministic
        - make caching safe
        - prevent magic numbers in physics codes
    """

    # redshift control
    z_max: float
    nz: int
    z_max_extended: float = None
    # High-z extension grid sizes (only used when z_max_extended is set).
    # 600 points gives <0.05% interpolation error for smooth H(z); increase to
    # 2000+ only if the model has sharp features above z ~ 3 (e.g. phase transitions).
    nz_extended: int = 600        # sparse high-z grid (z_max → 1100)
    nz_dense: int = 600           # dense low-z grid (0 → z_max)
    nz_dense_extended: int = 600  # dense high-z grid (z_max → 1100)

    # integration control
    integration_method: str = "quad"        # "quad" / "ode" / "trapz"
    rtol: float = 1e-8
    atol: float = 1e-10

    # interpolation control
    interp_kind: str = "linear"             # "linear" / "cubic"