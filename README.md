# COSMIX — COSmological Modular Inference eXplorer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19791571.svg)](https://doi.org/10.5281/zenodo.19791571)

COSMIX is a modular, likelihood-driven Bayesian inference framework for testing
modified gravity and dark energy cosmological models against observational data.
It supports multiple samplers (emcee, Dynesty, PolyChord) and a growing library
of likelihoods (CC, Pantheon+, DESI DR2 BAO, RSD, GW standard sirens, DES-SN5YR).

---

## Installation

### From PyPI

```bash
pip install pycosmix
```

### From source (for development)

```bash
git clone https://github.com/AmeyaKolhatkar/COSMIX.git
cd COSMIX
pip install -e .
```

For a specific tagged release (recommended for reproducible research):

```bash
git clone --branch v1.4.0 https://github.com/AmeyaKolhatkar/COSMIX.git
cd COSMIX
pip install -e .
```

Alternatively, a stable archived version is available via Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19791571.svg)](https://doi.org/10.5281/zenodo.19791571)

### Optional: build documentation locally

```bash
pip install -e ".[docs]"
```

### Optional: PolyChord nested sampler

Warning! The PolyChord nested sampler has not been tested extensively. If you still wish to use it, do

```bash
pip install -e src/cosmix/data/PolyChordLite
```

On **Windows** you will need [WinLibs/MinGW](https://winlibs.com/) with `gfortran`
and ensure `<mingw64>/bin` is on your PATH (or set `POLYCHORD_DLL_DIR`).

---

## Quick Start

### 1. Configure a run

Edit `input.yaml` (or copy it) to specify the model, datasets, sampler, and output options:

```yaml
run:
  name: "MyModel"
  run_id: "my_model_test_1"

model:
  name: fQ_EHybrid          # LCDM | fQ_Hybrid | fQ_EHybrid | fQ_LSR | fQ_LSR_IDE | fQ_Hybrid_IDE

likelihoods:
  - name: cosmicchronometers        # Cosmic Chronometers
  - name: desidr2bao                # DESI DR2 BAO
  - name: pantheonplus              # Pantheon+ (SN Ia)
  - name: rsd                       # Redshift Space Distortions

sampler:
  name: dynesty                     # emcee | dynesty | polychord
  init:
    nlive: 500

outputs:
  plots:
    trace: true
    corner: true
    residual: true
  archive: true
```

### 2. Run

Once installed, the `cosmix` command is available on your PATH:

```bash
cosmix input.yaml
```

Results are saved to `runs/<run_id>/` including the chain, diagnostics,
information criteria (AIC, BIC, DIC), and publication-quality figures.

---

## Project Structure

```
COSMIX/
├── pyproject.toml            # Package metadata, dependencies, build config
├── input.yaml                # Template run configuration
├── src/
│   └── cosmix/
│       ├── run_cosmix.py     # Main entry point — reads input.yaml and runs the pipeline
│       ├── Constants.py      # Physical constants (c, Omegar0)
│       ├── core/             # Framework internals (pipeline, parameter management, caching)
│       ├── theory/           # Cosmological models (LCDM, f(Q) variants)
│       │   ├── Solvers_/     # ODE/root solvers (RK4, analytical, Numba JIT)
│       │   └── Background_/  # Background state layout
│       ├── likelihoods/      # Observational likelihoods
│       ├── samplers/         # Sampler wrappers (emcee, Dynesty, PolyChord)
│       ├── drivers/          # Multi-chain convergence strategies
│       ├── postprocessing/   # Results, diagnostics, visualization, archival
│       │   └── Archive_/     # Run manifest, serialization (YAML/JSON/NumPy)
│       └── data/             # Observational data files
│           └── PolyChordLite/  # Bundled PolyChord source (build separately)
└── runs/                     # Output directory (gitignored contents)
```

---

## Available Models

| Key | Description |
|---|---|
| `LCDM` | Standard ΛCDM |
| `fQ_Hybrid` | f(Q) Hybrid model (analytical solution) |
| `fQ_EHybrid` | Extended f(Q) Hybrid model (ODE solver) |
| `fQ_Hybrid_Curved`| f(Q) Hybrid model with spatial curvature|
| `fQ_LSR` | f(Q) Log-Square-Root model |
| `fQ_LSR_IDE` | f(Q) LSR with Interacting Dark Energy |
| `fQ_Hybrid_IDE` | f(Q) Hybrid with Interacting Dark Energy |
| `fQ_Squared_Curved` | Squared f(Q) model with spatial curvature |

---

## Available Likelihoods

| Key | Dataset | Reference |
|---|---|---|
| `cosmicchronometers` | Cosmic Chronometers (correlated + uncorrelated) | Moresco et al. |
| `pantheonplus` | Pantheon+ (SN Ia, no SH0ES prior) | Brout et al. 2022 |
| `pantheonplusshoes` | Pantheon+SH0ES (SN Ia + H₀ prior) | Brout et al. 2022 |
| `desidr2bao` | DESI DR2 BAO (full GC combination) | DESI Collaboration 2025 |
| `rsd` | Redshift Space Distortions (fσ₈) | Various |
| `gw` | GW Standard Sirens (GW170817) | LIGO/Virgo |
| `desy5` | DES-SN5YR (SN Ia with probabilistic classification) | DES Collaboration |
| `desdovekie` | DES-Dovekie Likelihood | DES Collaboration |
| `shoes` | SH0ES H₀ Gaussian prior | Riess et al. |
| `trgb` | TRGB H₀ prior | Freedman et al. |
| `holicow` | H0LiCOW H₀ prior | Wong et al. |
| `compressedcmb`| Compressed CMB | Z. Zhai and Y. Wang |
| `egstatistic`| Eg Statistic | G. Alestas et al.|

---

## Adding a New Model

1. Create `src/cosmix/theory/MyModel.py` inheriting from `cosmix.core.CosmologyModelBase`.
2. Implement `declare_parameters()`, `check_physicality()`, and `get_requirements()`.
3. Register it through the `Registry.py` utility
   ```python
   from cosmix.core.Registry import cosmix_registry

   @cosmix_registry.register_model("MyModel")
   class MyModel(...)
   ```
4. Reference it in `input.yaml` as `model: { name: MyModel }`.

## Adding a New Likelihood

1. Create `src/cosmix/likelihoods/MyLikelihood.py` inheriting from `cosmix.core.LikelihoodBase_`.
2. Implement `declare_parameters()`, `get_requirements()`, and `lnlike()`.
3. Register it through the `Registry.py` utility
   ```python
   from cosmix.core.Registry import cosmix_registry

   @cosmix_registry.register_likelihood("MyLikelihood")
   class MyLikelihood(...)
   ```

---
## Under Development

- The PolyChord sampler has not been fully tested yet. User discretion is adviced.
- The `multi_auto` mode for `emcee` sampler is not fully developed. The goal is to implement the auto stop feature found in most Bayesian inference codes like `Cobaya`.
- The `MetropolisHastings.py` sampler is premature. Avoid using it for serious projects.
---

## Coming Soon
- Full `CAMB`-based CMB power spectrum likelihoods (preliminary interface already in place).

## Requirements

- Python ≥ 3.10
- See `pyproject.toml` for the full list of dependencies.
- A Fortran compiler (`gfortran`) is only needed if building PolyChordLite.

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See [LICENSE](LICENSE) for details.

---

## Citation

If you use COSMIX in your research, please cite the following:

```bibtex
@software{kolhatkar_cosmix_2026,
  author       = {Kolhatkar, Ameya},
  title        = {{COSMIX: Cosmological Modular Inference Explorer}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.19791571},
  url          = {https://doi.org/10.5281/zenodo.19791571}
}
```

---

## Contact
If there are any issues or suggestions for the code please feel free to contact me at \
kolhatkarameya1996@gmail.com

---
