# Background Config

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

    # integration control
    integration_method: str = "quad"        # "quad" / "ode" / "trapz"
    rtol: float = 1e-8
    atol: float = 1e-10

    # interpolation control
    interp_kind: str = "linear"             # "linear" / "cubic"