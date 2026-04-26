# Eg Statistic Likelihood

from pathlib import Path
from CORE_.LikelihoodBase_ import LikelihoodBase, GaussMargTerm
import numpy as np
import pandas as pd

Default_data_file = Path(__file__).resolve().parent.parent / "DATA_" / "Eg_statistic" / "eg_statistic.xlsx"

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