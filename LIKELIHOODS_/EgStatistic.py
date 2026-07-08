"""EgStatistic — E_G(z) statistic likelihood from galaxy surveys.

The E_G statistic is defined as:

    E_G(z) = Ω_{m0} / f(z)

where f(z) = d ln D / d ln a is the logarithmic growth rate.  Because
σ_80 cancels in the ratio, E_G constrains Ω_m and the growth-rate shape
but is insensitive to the amplitude σ_80.

Data file: DATA_/Eg_statistic/eg_statistic.xlsx
  Columns: z, Eg, Eg_err
"""

from pathlib import Path
from CORE_.Registry import cosmix_registry
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
import numpy as np
import pandas as pd

Default_data_file = Path(__file__).resolve().parent.parent / "DATA_" / "Eg_statistic" / "Eg_statistic_noVIPERS.xlsx"
@cosmix_registry.register_likelihood("egstatistic")
class EgStatistic(LikelihoodBase):
    name = "Eg"

    def __init__(self, pm, data_file=None):
        super().__init__(pm)
        if data_file is None:
            data_file = Default_data_file

        self.data_file = data_file

        data = pd.read_excel(self.data_file)
        self.z = data['z'].values
        self.Eg = data['Eg'].values
        self.Eg_err = data['Eg_err'].values

        self.data_size = len(self.z)
        self.produce_residuals = True

    def get_requirements(self):
        return {
            "f": self.z,
            "muG": self.z
        }
    
    def lnlike(self, theta, theory):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        f_model = theory.eval("f", self.z)
        muG_model = theory.eval("muG", self.z)

        Eg_model = Omegam0 * muG_model / f_model

        delta = self.Eg - Eg_model
        chi2 = np.sum( (delta/self.Eg_err)**2 )
        if not np.isfinite(chi2):
            return -np.inf
        
        return -0.5 * chi2
    
    def norm_term(self):
        return 0.0
    
    def get_theory_components(self, theta, theory, z_override=None):
        Omegam0 = self.pm.get_value(theta, "Omegam0")
        x = self.z if z_override is None else z_override
        d_vec = self.Eg
        f_model = theory.eval("f", x)
        muG_model = theory.eval("muG", x)
        th_vec = Omegam0 * muG_model / f_model
        sigma = self.Eg_err

        return {"Eg": (x, d_vec, th_vec, sigma)}
    
    def plot_constituents(self):
        return self.z, self.Eg, self.Eg_err

    def data_manifest(self):
        """Return one DataPoint per E_G measurement for overlap detection.

        Reads the survey label from the first available column named
        'Survey', 'survey', 'Data set', or 'Dataset' in the data file.
        Falls back to 'Unknown' if no such column exists.
        """
        from CORE_.OverlapChecker import DataPoint
        data = pd.read_excel(self.data_file)

        survey_col = next(
            (c for c in ("Survey", "survey", "Data set", "Dataset")
             if c in data.columns),
            None,
        )

        surveys = data[survey_col].values if survey_col else ["Unknown"] * len(self.z)

        manifest = []
        for i, z in enumerate(self.z):
            manifest.append(
                DataPoint(
                    likelihood_name=self.name,
                    observable="E_G",
                    z=float(z),
                    survey=str(surveys[i]),
                    value=float(self.Eg[i]),
                    error=float(self.Eg_err[i]),
                )
            )
        return manifest