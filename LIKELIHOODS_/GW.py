"""GW — gravitational-wave standard-siren distance likelihood.

Layered architecture (active layer set in the class body):

  Layer 0 — ideal standard sirens: Gaussian in d_L(z), no selection effects.
             Useful for architecture validation and H0 inference tests.

  Layer 1 — entry-level corrections: asymmetric distance posteriors from
             LVK summaries (log-normal or skewed likelihoods in d_L).

  Layer 2 — selection effects: detector-sensitivity-weighted likelihood
             normalization via LVK injection campaigns.  Introduces
             hyperparameters and Monte Carlo integrals.

  Layer 3 — dark sirens: statistical host identification using galaxy
             catalogs with completeness modeling and sky localization.

  Layer 4 — modified gravity: d_L^GW vs d_L^EM splitting, friction terms,
             running Planck mass; implemented via a GWKinematics engine layer.
"""
from CORE_.LikelihoodBase_ import LikelihoodBase
import numpy as np
import pandas as pd

from pathlib import Path as _Path
Default_data_file = _Path(__file__).resolve().parent.parent / "DATA_" / "GW" / "gwtc-1_data.xlsx"

class GWEvent:
    def __init__(self, name, z, dL, err_up, err_down):
        self.name = name
        self.z = float(z)
        self.dL = float(dL)
        self.err_up = float(err_up)
        self.err_down = float(err_down)


class GWStandardSiren(LikelihoodBase):
    name="GW_Standard_Siren"

    def __init__(self, pm, data_file=None, events=None):
        super().__init__(pm)

        if data_file is None:
            data_file = Default_data_file

        self.data_file = data_file
        
        if events is not None:
            self.events = events
        else:
            data = pd.read_excel(self.data_file)
            self.events = [
                GWEvent(
                    name=row["EVENT"],
                    z=row["z"],
                    dL=row["dL"],
                    err_up=row["dL_err_up"],
                    err_down=row["dL_err_down"]
                )
                for _, row in data.iterrows() if row["EVENT"] == "GW170817"
            ]

        self.data_size = len(self.events)
        self.produce_residuals = True

    
    def get_requirements(self):
        return {
            "dL": np.array([ev.z for ev in self.events])
        }
    
    def supports_marginalization(self, name):
        return False
    
    def marginalization_terms(self, theory):
        return None
    
    def lnlike(self, theta, theory):
        dL = theory.get("dL")["values"]
        self.use_asymmetric_errors = True

        lnL = 0.0
        for i, ev in enumerate(self.events):
            res = ev.dL - dL[i]
            if self.use_asymmetric_errors:
                sigma = ev.err_up if res > 0 else ev.err_down
            else:
                sigma = 0.5 * (ev.err_up + ev.err_down)
            
            if sigma <= 0: 
                return -np.inf
            lnL -= 0.5 * (res/sigma)**2

        return lnL  
    
    def get_theory_components(self, theta, theory, z_override=None):
        if z_override is None:
            x = np.array([ev.z for ev in self.events])
            d_vec = np.array([ev.dL for ev in self.events])
            th_vec = theory.eval("dL", x)

            err_up = np.array([ev.err_up for ev in self.events])
            err_down = np.array([ev.err_down for ev in self.events])

            return {"dL": (x, d_vec, th_vec, (err_up, err_down))}
        else:
            x = np.asarray(z_override)
            th_vec = theory.eval("dL", x)

            return {"dL": (x, None, th_vec, None)}  


            
