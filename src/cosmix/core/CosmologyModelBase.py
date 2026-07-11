"""
CosmologyModelBase — abstract base class for all COSMIX cosmological models.

A concrete model must implement three methods:

    background_problem(theta, z_grid)
        Returns a BackgroundProblem descriptor (AnalyticalProblem or
        NumbaODEProblem).  BackgroundKinematics calls problem.solve() to
        obtain H(z) and any extra derived quantities.

    background_config()
        Returns a BackgroundConfig specifying the redshift range and
        grid resolution appropriate for this model.

    declare_parameters()  [classmethod]
        Returns a list of Parameter objects for this model's free and
        fixed parameters.  Called once during Pipeline construction.

Optionally override:
    check_physicality(theta)   — return False to reject unphysical points
                                 before any expensive computation.
    muG(z, theta, bg_engine)   — gravitational coupling μ_G(z) for the
                                 growth equation; default is GR (μ_G = 1).
"""
import numpy as np
from abc import ABC, abstractmethod
from cosmix.core.BackgroundKinematics import BackgroundKinematics
from cosmix.core.GrowthKinematics import GrowthKinematics
from dataclasses import replace
from cosmix.core.TheoryCache import TheoryCache
from cosmix.core.EngineResolver import fill_theory_cache

# ══════════════════════════════════════════════════════════════════════════════
# CosmologyModelBase
# ══════════════════════════════════════════════════════════════════════════════
class CosmologyModelBase(ABC):
    name = "base"

    def __init__(self, parameter_manager):
        self.pm = parameter_manager

    def check_physicality(self, theta):
        """
        Optional parameter checks where the model enters unphysical conditions.
        """
        return True

    def compute_theory(self, theta, requirements):          # This method is called by the sampler
        """
        theta : array-like free parameter vector
        requirements : dict mapping from observable name --> redshift array

        returns TheoryCache
        """
        theory = TheoryCache()

        H0 = self.pm.get_value(theta, "H0")
        if H0 <= 0.0:
            theory.mark_invalid()
            return theory
        
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        if Omegam0 <= 0.0 or Omegam0 >= 1.0:
            theory.mark_invalid()
            return theory
        
        if not self.check_physicality(theta):
            theory.mark_invalid()
            return theory

        try:
            engines = self.build_engines(theta, requirements)
            return fill_theory_cache(theory, requirements, engines)
        except Exception as e:
            #print(f"[CosmologyModelBase] PIPELINE CRASH CAUGHT: {e}")
            theory.mark_invalid()
            return theory

    @abstractmethod
    def background_problem(self, theta, z_grid):
        """
        Returns a BackgroundProblem descriptor (AnalyticalProblem, NumbaODEProblem, or ImplicitProblem).
        BackgroundKinematics calls problem.solve(z_grid) to get {"H": ...} and optional extras.
        """
        pass

    @abstractmethod
    def background_config(self):
        """
        Returns a BackgroundConfig instance appropriate for this model.
        """
        pass

    def build_engines(self, theta, requirements):
        """
        Override in concrete models.
        Must return a list of engines.
        """
        engines = []

        base_config = self.background_config()

        needs_growth = True if any(name in ("delta", "f", "fsigma8") for name in requirements) else False
        # Trigger the extended z grid whenever any observable is requested
        # at z beyond the base config's z_max (e.g. CompressedCMB requests H up to z~1100).
        needs_high_z = any(
            len(req.get("z", [])) > 0 and np.asarray(req["z"]).max() > base_config.z_max
            for req in requirements.values()
        )

        if needs_growth or needs_high_z:
            active_config = replace(base_config, z_max_extended=1100.0)
        else:
            active_config = base_config

        bk = BackgroundKinematics(
            model=self,
            theta=theta,
            config=active_config
        )
        engines.append(bk)

        if needs_growth:
            gk = GrowthKinematics(
                background=bk,
                model=self,
                theta=theta
            )
            engines.append(gk)

        needs_camb = any(
            name in ("Pk_linear", "Pk_nonlinear", "sigma8_camb", "r_drag")
            for name in requirements
        )
        if needs_camb:
            from cosmix.core.CAMBKinematics import CAMBKinematics
            # Collect z values from all requirements for the CAMB z-grid
            camb_z = np.unique(np.concatenate([
                req.get("z", np.array([]))
                for req in requirements.values()
            ]))
            ck = CAMBKinematics(
                model=self,
                theta=theta,
                z_grid=camb_z
            )
            engines.append(ck)

        return engines 


    @classmethod
    @abstractmethod
    def declare_parameters(cls):
        pass

    def plot_constituents(self):
        """
        Optional method to call the data points for a specific likelihood, for plotting purposes. 
        Should return : z, qty, qty_err 
        """
        pass 

    def muG(self, z, theta, bg_engine):
        """
        Gravitational coupling mu_G(z) for the growth equation.
        Default: GR limit (mu_G = 1). Override in modified gravity models.
        """
        return np.ones_like(np.asarray(z, dtype=float))
    
def make_free(model_cls, *param_names):
    """Return a copy of model_cls with the named parameters set to free."""
    from cosmix.core.ParameterManager_ import Parameter, UniformPrior

    def declare_parameters(cls):
        result = []
        for p in model_cls.declare_parameters():
            if p.name in param_names:
                p = Parameter(name=p.name, latex=p.latex, prior=p.prior,
                              role=p.role, status="free",
                              value=p.value, proposed_scale=p.proposed_scale)
            result.append(p)
        return result

    return type(
        f"{model_cls.__name__}_free_{'_'.join(param_names)}",
        (model_cls,),
        {"declare_parameters": classmethod(declare_parameters),
         "name": f"{model_cls.name}+{'_'.join(param_names)}"}
    )


def make_fixed(model_cls, *param_names):
    """Return a copy of model_cls with the named parameters set to fixed (at their declared default values)."""
    from cosmix.core.ParameterManager_ import Parameter

    def declare_parameters(cls):
        result = []
        for p in model_cls.declare_parameters():
            if p.name in param_names:
                p = Parameter(name=p.name, latex=p.latex, prior=p.prior,
                              role=p.role, status="fixed",
                              value=p.value, proposed_scale=p.proposed_scale)
            result.append(p)
        return result

    return type(
        f"{model_cls.__name__}_fixed_{'_'.join(param_names)}",
        (model_cls,),
        {"declare_parameters": classmethod(declare_parameters),
         "name": f"{model_cls.name}_fixed_{'_'.join(param_names)}"}
    )