# COSMIX — COSmological Modular Inference eXplorer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19791571.svg)](https://doi.org/10.5281/zenodo.19791571)

COSMIX is a modular, likelihood-driven Bayesian inference framework for testing
modified gravity and dark energy cosmological models against observational data.
It supports multiple samplers (emcee, Dynesty, PolyChord) and a growing library
of likelihoods (CC, Pantheon+, DESI DR2 BAO, RSD, GW standard sirens, DES-SN5YR).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AmeyaKolhatkar/COSMIX.git
cd COSMIX
```

For a specific tagged release (recommended for reproducible research):

```bash
git clone --branch v1.0.0 https://github.com/AmeyaKolhatkar/COSMIX.git
cd COSMIX
```

Alternatively, a stable archived version is available via Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19791571.svg)](https://doi.org/10.5281/zenodo.19791571)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For the PolyChord nested sampler (optional, requires a Fortran compiler):

```bash
pip install -e DATA_/PolyChordLite
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
  - name: CC                 # Cosmic Chronometers
  - name: DDTB               # DESI DR2 BAO
  - name: PP                 # Pantheon+ (SN Ia)
  - name: RSD                # Redshift Space Distortions

sampler:
  name: dynesty              # emcee | dynesty | polychord
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

```bash
python run_cosmix.py input.yaml
```

Results are saved to `RUNS_/<run_id>/` including the chain, diagnostics,
information criteria (AIC, BIC, DIC), and publication-quality figures.

---

## Project Structure

```
COSMIX/
├── run_cosmix.py         # Main entry point — reads input.yaml and runs the pipeline
├── Constants.py          # Physical constants (c, Omegar0)
├── input.yaml            # Template run configuration
│
├── CORE_/                # Framework internals (pipeline, parameter management, caching)
├── THEORY_/              # Cosmological models (LCDM, f(Q) variants)
│   └── Solvers_/         # ODE/root solvers (RK4, analytical, Numba JIT)
├── LIKELIHOODS_/         # Observational likelihoods
├── SAMPLERS_/            # Sampler wrappers (emcee, Dynesty, PolyChord)
├── DRIVERS_/             # Multi-chain convergence strategies
├── POST_PROCESSING_/     # Results, diagnostics, visualization, archival
│   └── Archive_/         # Run manifest, serialization (YAML/JSON/NumPy)
├── DATA_/                # Observational data files
│   └── PolyChordLite/    # Bundled PolyChord source (build separately)
├── RUNS_/                # Output directory (gitignored contents)
```

---

## Available Models

| Key | Description |
|---|---|
| `LCDM` | Standard ΛCDM |
| `fQ_Hybrid` | f(Q) Hybrid model (analytical solution) |
| `fQ_EHybrid` | Extended f(Q) Hybrid model (ODE solver) |
| `fQ_LSR` | f(Q) Log-Square-Root model |
| `fQ_LSR_IDE` | f(Q) LSR with Interacting Dark Energy |
| `fQ_Hybrid_IDE` | f(Q) Hybrid with Interacting Dark Energy |

---

## Available Likelihoods

| Key | Dataset | Reference |
|---|---|---|
| `CC` | Cosmic Chronometers (correlated + uncorrelated) | Moresco et al. |
| `PP` | Pantheon+ (SN Ia, no SH0ES prior) | Brout et al. 2022 |
| `PPS` | Pantheon+SH0ES (SN Ia + H₀ prior) | Brout et al. 2022 |
| `DDTB` | DESI DR2 BAO (full GC combination) | DESI Collaboration 2025 |
| `RSD` | Redshift Space Distortions (fσ₈) | Various |
| `GW` | GW Standard Sirens (GW170817) | LIGO/Virgo |
| `D5` | DES-SN5YR (SN Ia with probabilistic classification) | DES Collaboration |
| `SH0ES` | SH0ES H₀ Gaussian prior | Riess et al. |
| `TRGB` | TRGB H₀ prior | Freedman et al. |
| `H0LiCOW` | H0LiCOW H₀ prior | Wong et al. |
| `CompCMB`| Compressed CMB | Z. Zhai and Y. Wang |
| `Eg`| Eg Statistic | G. Alestas et al.|

---

## Adding a New Model

1. Create `THEORY_/MyModel.py` inheriting from `CORE_.CosmologyModelBase`.
2. Implement `declare_parameters()`, `check_physicality()`, and `get_requirements()`.
3. Register it in `run_cosmix.py` (under `MODEL_REGISTRY`):
   ```python
   from THEORY_.MyModel import MyModel
   MODEL_REGISTRY["MyModel"] = MyModel
   ```
4. Reference it in `input.yaml` as `model: { name: MyModel }`.

## Adding a New Likelihood

1. Create `LIKELIHOODS_/MyLikelihood.py` inheriting from `CORE_.LikelihoodBase_`.
2. Implement `declare_parameters()`, `get_requirements()`, and `lnlike()`.
3. Register it in `run_cosmix.py` under `LIKELIHOOD_REGISTRY`.

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for the full list of packages.
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
