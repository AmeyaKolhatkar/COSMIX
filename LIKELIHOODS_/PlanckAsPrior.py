"""PlanckAsPrior — Gaussian prior on the primordial amplitude A_s.

Since σ₈₀² ∝ A_s (for fixed transfer function shape), a Gaussian prior on
ln(10¹⁰ A_s) is equivalent to a Gaussian prior on 2·ln(σ₈₀/σ₈₀_ref) with
the same width.  This likelihood implements that constraint directly on the
free σ₈₀ parameter, given a calibration reference σ₈₀_ref from a companion
ΛCDM run with the same dataset.

Reference
---------
Planck 2018 results VI (arXiv:1807.06209), Table 1:
    ln(10¹⁰ A_s) = 3.044 ± 0.014   (TT,TE,EE + lowE)

Usage
-----
In input.yaml::

    likelihoods:
      - name: PlanckAs
        class: PlanckAsPrior
        options:
          sigma80_ref: 0.801   # posterior mean σ₈₀ from ΛCDM run on same dataset

If ``sigma80_ref`` is omitted, the Planck 2018 CMB-only value (0.794) is used
and a RuntimeWarning is issued.  Always supply the value from your companion
ΛCDM run for self-consistency.
"""

import warnings
import numpy as np
from CORE_.LikelihoodBase_ import LikelihoodBase
from CORE_.ParameterManager_ import Parameter, UniformPrior


class PlanckAsPrior(LikelihoodBase):
    """Gaussian prior on ln(10¹⁰ A_s), applied through σ₈₀.

    Parameters
    ----------
    pm : ParameterManager
    sigma80_ref : float
        ΛCDM σ₈₀ calibration value from a companion run on the same dataset.
        Defaults to the Planck 2018 CMB-only value (0.811), but using your
        own ΛCDM posterior mean is more self-consistent.
    lnAs_mean : float
        Planck central value of ln(10¹⁰ A_s).  Default: 3.044.
    lnAs_sigma : float
        Planck 1σ uncertainty on ln(10¹⁰ A_s).  Default: 0.014.
    """

    name = "PlanckAs"

    #: Sentinel so we can detect when the user has not supplied sigma80_ref.
    _SIGMA80_REF_DEFAULT = 0.794

    def __init__(self, pm,
                 sigma80_ref: float = None,
                 lnAs_mean:   float = 3.044,
                 lnAs_sigma:  float = 0.014):
        super().__init__(pm)
        if sigma80_ref is None:
            warnings.warn(
                "PlanckAsPrior: sigma80_ref not set in input.yaml — "
                f"falling back to default ({self._SIGMA80_REF_DEFAULT}). "
                "Set options: {{sigma80_ref: <value>}} to the σ₈₀ posterior "
                "mean from your companion ΛCDM run on the same dataset.",
                RuntimeWarning, stacklevel=2,
            )
            sigma80_ref = self._SIGMA80_REF_DEFAULT
        self.sigma80_ref = sigma80_ref
        self.lnAs_mean   = lnAs_mean
        self.lnAs_sigma  = lnAs_sigma
        self.data_size   = 1
        self.produce_residuals = False

    @classmethod
    def declare_parameters(cls):
        """Declare sigma80 as a free nuisance parameter.

        PlanckAsPrior is the logical owner of sigma80 because it implements
        the A_s prior expressed through sigma80.  Other likelihoods (e.g.
        SDSSDR16BAO) that also use sigma80 should rely on Pipeline
        deduplication rather than re-declaring it.
        """
        return [
            Parameter(
                name="sigma80",
                latex=r"\sigma_{80}",
                prior=UniformPrior(0.6, 1.2),
                role="nuisance",
                status="free",
                value=0.8,
                proposed_scale=0.01,
            )
        ]

    def get_requirements(self):
        return {}

    def lnlike(self, theta, theory):
        sigma80 = self.pm.get_value(theta, "sigma80")
        if sigma80 <= 0.0:
            return -np.inf

        # ln(A_s_implied / A_s_ref) = 2 * ln(sigma80 / sigma80_ref)
        lnAs_implied = self.lnAs_mean + 2.0 * np.log(sigma80 / self.sigma80_ref)
        chi2 = ((lnAs_implied - self.lnAs_mean) / self.lnAs_sigma) ** 2
        return -0.5 * chi2

    def norm_term(self):
        return 0.0
