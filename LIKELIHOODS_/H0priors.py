"""H0priors — simple Gaussian H₀ priors as standalone likelihoods.

Each class encodes a single measurement of the Hubble constant as a
Gaussian likelihood.  They plug into the pipeline exactly like any
other likelihood and can be combined freely with distance or growth data.

Available priors
----------------
SH0ESprior   H₀ = 74.22 ± 1.82 km/s/Mpc  (Riess et al. 2019, LMC Cepheids)
TRGBprior    H₀ = 69.8  ± 1.9  km/s/Mpc  (Freedman et al. 2019, TRGB)
H0LiCOWprior H₀ = 73.3  ± 1.75 km/s/Mpc  (Wong et al. 2020, strong lensing)
"""
from CORE_.LikelihoodBase_ import LikelihoodBase

"""class GaussianH0Prior(LikelihoodBase):

    def __init__(self, pm, name, mean, sigma):
        super().__init__(pm)
        self.name = name
        self.mean = mean
        self.sigma = sigma

    def lnlike(self, theta, theory):
        H0 = self.pm.get_value(theta, "H0")
        return -0.5*((H0-self.mean)/self.sigma)**2


SH0ES = GaussianH0Prior(pm,"SH0ES",74.22,1.82)
TRGB  = GaussianH0Prior(pm,"TRGB",69.8,1.9)
H0LiCOW = GaussianH0Prior(pm,"H0LiCOW",73.3,1.75)"""


"""
1. SH0ES 

    Using H0 = 74.22 +- 1.82 km/s/Mpc from 

    "Large Magellanic Cloud Cepheid Standards Provide a 1% Foundation for the Determination of the Hubble Constant 
    and Stronger Evidence for Physics Beyond ΛCDM", Adam G. Riess, Stefano Casertano, Wenlong Yuan, Lucas M. Macri
    and Dan Scolnic4
"""


class SH0ESprior(LikelihoodBase):
    name="SH0ES"

    def __init__(self, pm):
        super().__init__(pm)
        self.H0_mean = 74.22
        self.sigma = 1.82
        self.data_size = 1
        self.produce_residuals = False

    def lnlike(self, theta, theory):
        H0 = self.pm.get_value(theta, "H0")
        chi2 = ( (H0 - self.H0_mean) / self.sigma )**2

        return -0.5*chi2
    
    def norm_term(self):
        return 0.0
    
    def get_requirements(self):
        return {}
    
    
"""
2. TRGB 

    Using H0 = 69.8 +- 1.9 km/s/Mpc from 

    "The Carnegie-Chicago Hubble Program. VIII. An Independent Determination of the Hubble Constant Based on the
    Tip of the Red Giant Branch", Wendy L. Freedman, Barry F. Madore, Dylan Hatt, 1 Taylor J. Hoyt, In Sung Jang,
    Rachael L. Beaton, Christopher R. Burns, Myung Gyoon Lee, Andrew J. Monson, Jillian R. Neeley, M. M. Phillips,
    Jeffrey A. Rich and Mark Seibert
"""
class TRGBprior(LikelihoodBase):
    name="TRGB"

    def __init__(self, pm):
        super().__init__(pm)
        self.H0_mean = 69.8
        self.sigma = 1.9
        self.data_size = 1
        self.produce_residuals = False

    def lnlike(self, theta, theory):
        H0 = self.pm.get_value(theta, "H0")
        chi2 = ( (H0 - self.H0_mean) / self.sigma )**2

        return -0.5*chi2
    
    def norm_term(self):
        return 0.0
    
    def get_requirements(self):
        return {}
    


"""
3. H0LiCOW 

    Using H0 = 73.3 + 1.7 - 1.8 km/s/Mpc from 

    "H0LiCOW XIII. A 2.4% measurement of H0 from lensed quasars: 5.3σ tension between early and late-Universe probes",
    Kenneth C. Wong, Sherry H. Suyu, Geoff C.-F. Chen, Cristian E. Rusu, Martin Millon, Dominique Sluse, Vivien Bonvin,
    Christopher D. Fassnacht, Stefan Taubenberger, Matthew W. Auger, Simon Birrer, James H. H. Chan, Frederic Courbin,
    Stefan Hilbert, Olga Tihhonova, Tommaso Treu, Adriano Agnello, Xuheng Ding, Inh Jee, Eiichiro Komatsu, Anowar J. Shajib,
    Alessandro Sonnenfeld, Roger D. Blandford, L´eon V. E. Koopmans, Philip J. Marshall and Georges Meylan
"""
class H0LiCOWPrior(LikelihoodBase):
    name="H0LiCOW"

    def __init__(self, pm):
        super().__init__(pm)
        self.H0_mean = 73.3
        self.sigma = 1.75
        self.data_size = 1
        self.produce_residuals = False

    def lnlike(self, theta, theory):
        H0 = self.pm.get_value(theta, "H0")
        chi2 = ( (H0 - self.H0_mean) / self.sigma )**2

        return -0.5*chi2
    
    def norm_term(self):
        return 0.0
    
    def get_requirements(self):
        return {}


    
