# Learned Lagrangian LES

**This code contains:**
* SpectralDNS solver in JAX, with example configs:
    * 2D/3D decaying Taylor-Green vortex
    * 2D Kolmogorov flow, see [JAX-CFD](https://www.pnas.org/content/118/21/e2101784118)
    * 3D Homogeneous Isotropic Turbulence (HIT) with or without forcing
* Utilities to relax Smoothed Particle Hydrodynamics (SPH) particles based on JAX-SPH
* Utilities to evolve SPH particles in parallel to the SpectralDNS solver
* Utilities to generate datasets compatible with LagrangeBench

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
