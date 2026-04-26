# GW Likelihood
"""
==================================================================
LAYER 0
==================================================================

Ideal standard sirens
    similar to implementing SNeIa data using dL(z) without host corrections.

This ignores:
    - Selection effects
    - Population distributions
    - Detector sensitivity
    - Inclination bias
    - Malmquist bias

Useful for:
    - Validating architecture, distance plumbing, and H0 inference.

==================================================================
LAYER 1
==================================================================

Entry level bias corrections

The following things get added
    - Use LVK provided posterior summaries like - asymmetric distance errors nad marginalized distance posteriors
    - Possibly log-normal or skewed likelihoods in dL

==================================================================
LAYER 2
==================================================================

Selection Effects (ADVANCED)

The selection function modifies the likelihood. The function itself depends on
    - detector sensitiity,
    - population model,
    - cosmology.

This introduces 
    - Hyperparameters
    - Monte Carlo Integrals
    - Strong Priors

Implementation
    - as a normalization term
    - cached and reused
    - using LVK injection campaigns

==================================================================
LAYER 3
==================================================================

Dark Sirens (Statistical Hosts)

Requires
    - Galaxy Catalogs
    - completeness modeling
    - sky localization

==================================================================
LAYER 4
==================================================================

Modified Gravity effects

Introduction of 
    - dL_GW vs dL_EM
    - friction terms
    - running Planck mass
    - theory dependent propagation

Will add 
    - a new GWKinematics layer without changing the likelihood structure and the selection formalism

"""

#------------------------------
# Preamble 
#------------------------------
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


            
