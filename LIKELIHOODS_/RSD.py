"""RedshiftSpaceDistortion — f σ₈ likelihood from galaxy surveys.

Measures the growth rate of large-scale structure via the combination
f σ₈(z), where f = d ln D / d ln a is the logarithmic growth rate
and σ₈(z) = σ₈₀ × D(z)/D(0) is the matter fluctuation amplitude.

The Alcock-Paczynski correction is applied when fiducial H(z) and D_M(z)
values are provided in the data file, giving the corrected quantity:

    (f σ₈)_obs × [H_fid(z) D_M_fid(z)] / [H_th(z) D_M_th(z)]

Data file: DATA_/RSD/RSD-1.xlsx  (columns: z, fs8, fs8_err, fidom)
nuisance parameter: sigma80 (σ₈ today), uniform prior [0.6, 1.2].
"""
import numpy as np
import pandas as pd
from CORE_.LikelihoodBase_ import LikelihoodBase
from CORE_.ParameterManager_ import Parameter, UniformPrior, GaussianPrior
from THEORY_.LCDM_ import LCDM
from CORE_.BackgroundKinematics import BackgroundKinematics
from CORE_.BackgroundConfiguration import BackgroundConfig
from CORE_.ParameterManager_ import ParameterManager

from pathlib import Path as _Path
Default_data_file = _Path(__file__).resolve().parent.parent / "DATA_" / "RSD" / "RSD-1.xlsx"

class RedshiftSpaceDistortion(LikelihoodBase):
    name = "RSD"

    def __init__(self, pm, data_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file

        self.data_file = data_file

        data = pd.read_excel(self.data_file)
        self.z = data['z'].values
        self.fs8 = data['fs8'].values
        self.fs8_err = data['fs8_err'].values
        self.omega_fid = data['fidom'].values

        self.H_fid, self.DM_fid = self._fiducial_geometry()

        self.data_size = len(self.z)
        self.produce_residuals = True

    @classmethod
    def declare_parameters(cls):
        return [
            Parameter(
                name="sigma80",
                latex=r"\sigma_{80}",
                prior=UniformPrior(0.6, 1.2),
                role="nuisance",
                status="free",
                proposed_scale=0.01
            )
        ]

    def get_requirements(self):
        return {
            "fsigma8": self.z,
            "H": self.z,
            "DM": self.z
        }
    
    def _fiducial_geometry(self):
        z = self.z
        Om0_fid = self.omega_fid

        H_fid = np.empty_like(z)
        DM_fid = np.empty_like(z)

        def build_fixed_pm(model_cls, fixed_values):
            PM = ParameterManager()
            for p in model_cls.declare_parameters():
                if p.name in fixed_values:
                    p_fixed = Parameter(
                        name=p.name,
                        latex=p.latex,
                        prior=p.prior,
                        role=p.role,
                        status="fixed",
                        value=fixed_values[p.name]
                    )
                    PM.add_parameter(p_fixed)

                else:
                    PM.add_parameter(p)
            
            PM.freeze()

            return PM
        
        unique_oms = np.unique(Om0_fid)

        for om in unique_oms:
            pm_fid = build_fixed_pm( 
                LCDM, 
                {"H0": 70.0, "Omegam0": float(om)})
            model_fid = LCDM(pm_fid)            # LCDM instantiation
            bk = BackgroundKinematics(
                model=model_fid,
                theta=np.array([]),             # no free parameters
                config=BackgroundConfig(z_max=np.max(z)+0.1, nz=2000)
            )

            mask = np.isclose(Om0_fid, om)
            z_sub = z[mask]
            H_fid[mask] = bk.H(z_sub)
            DM_fid[mask] = bk.DM(z_sub)

        return H_fid, DM_fid

    def lnlike(self, theta, theory):
        fs8_model = theory.eval("fsigma8", self.z)
        H_model = theory.eval("H", self.z)
        DM_model = theory.eval("DM", self.z)
        
        AP = (self.H_fid * self.DM_fid) / (H_model * DM_model)

        fs8_corrected = AP * fs8_model
        delta = self.fs8 - fs8_corrected
        chi2 = np.sum( (delta / self.fs8_err)**2 )
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5*chi2

    def norm_term(self):
        return 0.0
    
    def get_theory_components(self, theta, theory, z_override=None):
        x = self.z if z_override is None else z_override
        d_vec = self.fs8
        th_vec = theory.eval("fsigma8", x)
        sigma = self.fs8_err

        return {"fsigma8": (x, d_vec, th_vec, sigma)}
    
    def get_plots(self):
        return {
            "fsigma8": True
        }
    
    def plot_constituents(self):
        return self.z, self.fs8, self.fs8_err