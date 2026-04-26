# Cosmic chronometers Covariance

<img align="right" src="figures/Hz_CC.png" alt="drawing" width="60%">

A primer on how to estimate cosmic chronometers covariance, how to decouple it into its various components, and how to apply it for cosmological calculations

+ Dependencies:
   + **Python 3**
   + **`numpy`**
   + **`scipy`**
   + **`matplotlib`**
   + **`emcee`**

#### General introduction
Following <a href="https://ui.adsabs.harvard.edu/abs/2020ApJ...898...82M/abstract">Moresco et al. (2020)</a>, cosmic chronometers (CC) covariance is defined as the combination of a statistical and systematic part:

```math
{\rm Cov}_{ij}= {\rm Cov}_{ij}^{\rm stat}+ {\rm Cov}_{ij}^{\rm syst}
```
where $`{\rm Cov}_{ij}^{\rm syst}`$, for simplicity and transparency, is decomposed in several contributions:

```math
{\rm Cov}_{ij}^{\rm syst}= {\rm Cov}_{ij}^{\rm met}+ {\rm Cov}_{ij}^{\rm young}+ {\rm Cov}_{ij}^{\rm model}
```

where:
- $`{\rm Cov}_{ij}^{\rm met}`$ is the contribution to the covariance matrix due to uncertainty in the estimate fo the stellar metallicity;
- $`{\rm Cov}_{ij}^{\rm young}`$ is the part of the covariance matrix affected by an eventual residual young component in galaxy spectra (see <a href="https://ui.adsabs.harvard.edu/abs/2018ApJ...868...84M/abstract">Moresco et al. (2018)</a>);
- $`{\rm Cov}_{ij}^{\rm model}`$ is the contribution to the covariance matrix arising from modelling, that, in turn, can be decomposed in:
```math
{\rm Cov}_{ij}^{\rm model}={\rm Cov}_{ij}^{\rm SFH}+{\rm Cov}_{ij}^{\rm IMF}+{\rm Cov}_{ij}^{\rm st. lib.}+{\rm Cov}_{ij}^{\rm SPS}
```
where:
- $`{\rm Cov}_{ij}^{\rm SFH}`$ is the contribution to the model covariance matrix due to uncertainty in star formation history;
- $`{\rm Cov}_{ij}^{\rm IMF}`$ is the contribution to the model covariance matrix due to uncertainty in the IMF adopted;
- $`{\rm Cov}_{ij}^{\rm st. lib.}`$ is the contribution to the model covariance matrix due to uncertainty in the stellar library adopted;
- $`{\rm Cov}_{ij}^{\rm SPS}`$ is the contribution to the model covariance matrix due to uncertainty in the stellar population synthesis model adopted.

Amongst these terms, $`{\rm Cov}_{ij}^{\rm met}`$ and $`{\rm Cov}_{ij}^{\rm young}`$ are purely diagonal terms, since they are related to the estimate of physical property of a galaxy (the stellar metallicity, and the eventual contamination by a younger subdominant population) uncorrelated from bin to bin. $`{\rm Cov}_{ij}^{\rm model}`$, instead, has been conservatively estimated as the contribution from different redshifts are fully correlated. The full systematic covariance matrix $`{\rm Cov}_{ij}^{\rm syst}`$ is, however, invertible.

#### Practical information
In folder '*/examples/*' can be found different jupyter notebooks showing how to estimate CC covariance, and how to use it for cosmological calculations.

- in **CC_covariance.ipynb**, how to estimate CC covariance with current dataset, with the corresponding plot;
- in **CC_covariance_components.ipynb**, how to split the covariance matrix into the various components discussed above for the data in <a href="https://ui.adsabs.harvard.edu/abs/2012JCAP...08..006M/abstract">[1]</a>, <a href="https://ui.adsabs.harvard.edu/abs/2015MNRAS.450L..16M/abstract">[2]</a>, and <a href="https://ui.adsabs.harvard.edu/abs/2016JCAP...05..014M/abstract">[3]</a>;
- in **CC_fit.ipynb**, how to use the covariance of current data for a minimal cosmological fit, with a fLCDM model (using `emcee`).


#### Acknowledgements
If using the CC data, please remember to cite the original papers providing the measurements:
- <a href="https://ui.adsabs.harvard.edu/abs/2012JCAP...08..006M/abstract">Moresco et al. (2012)</a>;
- <a href="https://ui.adsabs.harvard.edu/abs/2015MNRAS.450L..16M/abstract">Moresco et al. (2015)</a>;
- <a href="https://ui.adsabs.harvard.edu/abs/2016JCAP...05..014M/abstract">Moresco et al. (2016)</a>.

