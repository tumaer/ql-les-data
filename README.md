# Dataset Generation Code for *Data-Driven Discretizations of Quasi-Lagrangian Turbulence*

**This code contains:**
* SpectralDNS solver in JAX, with example configs:
    * Decaying Taylor-Green vortex in 2D from TVF-SPH [(Adami et al., 2013)](https://www.sciencedirect.com/science/article/abs/pii/S002199911300096X) and in 3D from [Brachet et al. (1983)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/smallscale-structure-of-the-taylorgreen-vortex/5C32D7A4CDF8E2A200FF62A046BC2F5B).
    * 2D Kolmogorov flow, see JAX-CFD [(Kochkov et al., 2021)](https://www.pnas.org/content/118/21/e2101784118).
    * 3D Homogeneous Isotropic Turbulence (HIT) with forcing (see [Lamorgese et al. (2005)](https://pubs.aip.org/aip/pof/article/17/1/015106/917179/Direct-numerical-simulation-of-homogeneous) and [Mortensen and Langtangen (2016)](https://arxiv.org/abs/1602.03638v1)) and without forcing following [Tian et al. (2023)](https://www.pnas.org/doi/10.1073/pnas.2213638120). Initialization follows [Rogallo (1981)](https://ntrs.nasa.gov/api/citations/19810022965/downloads/19810022965.pdf).
* Utilities to relax Smoothed Particle Hydrodynamics (SPH) particles based on JAX-SPH [(Toshev et al., 2024)](https://arxiv.org/abs/2403.04750).
* Utilities to evolve SPH particles in parallel to the spectralDNS solver [(Mortensen and Langtangen, 2016)](https://arxiv.org/abs/1602.03638v1).
* Utilities to generate datasets compatible with LagrangeBench [(Toshev et al., 2023)](https://arxiv.org/abs/2309.16342).

## Install

```bash
poetry install
source .venv/bin/activate
pip install "jax[cuda12]==0.4.29"
```

## Getting Started

* Validation run with reference kinetic energy after 10 steps.
```bash
python main.py config=configs/tgv_validate.yaml
```

* To generate an HIT dataset with $32^3$ particles, we run spectral DNS on a $256^3$ grid and spectrally coarsen/filter to $32^3$.
```bash
python main.py config=configs/hit.yaml
```

## Datasets

The two datasets used in the paper can be regenerated with the following scripts:
* `sbatch gen_dataset/slurm_kolm2d_64_1.sh` - 80 min/traj x 20 trajs
* `sbatch gen_dataset/slurm_hit3d_32_1.sh` - 60 min/traj x 20 trajs
> At the bottom of these scripts are the commands to convert the simulations into a dataset file.
