# Learned Lagrangian LES

**This code contains:**

- [x] Spectral HIT solver. Based on https://github.com/spectralDNS/spectralDNS
- [ ] Set up case
    - [ ] CBC initial conditions
    - [ ] forced turbulence
    - [ ] enforce Re_lambda, see https://www.sto.nato.int/publications/AGARD/AGARD-AR-345/AGARD-AR-345.pdf
- [x] Evolution of SPH particles following spectral dynamics
    - [x] start integrating from a relaxed state
    - [ ] integrate solver into particle evolution and make smaller time steps, e.g. 0.1dt
- [ ] Dataset generation utils
- [ ] Evaluation metrics
    - [x] Jonas' MLS interpolation
    - [x] direct DFT on points to compute spectrum
    - [ ] density evaluation

## Install

```bash
poetry install
source .venv/bin/activate
pip install "jax[cuda12]==0.4.29"
```

## Run

Validation run with reference kinetic energy after 10 steps.

```bash
python main.py config=configs/tgv_validate.yaml
```

To generate a dataset of 32^3 particles, we run spectral DNS on 128^3 and spectrally coarsen to 32^3.

```bash
python main.py config=configs/hit.yaml
```

For higher quality, run the following.

```bash
nohup bash run.sh >> hit_192_5_0002.out 2>&1 &
```
