# Learned Lagrangian Large Eddy Simulation

**This code contains:**

- [x] Spectral HIT solver. Based on  https://github.com/spectralDNS/spectralDNS
- [ ] Evolution of SPH particles following spectral dynamics
- [ ] Dataset generation utils
- [ ] Evaluation metrics

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
