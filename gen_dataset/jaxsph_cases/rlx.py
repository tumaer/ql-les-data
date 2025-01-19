"""Relax particles in a periodic box."""


import jax.numpy as jnp
import numpy as np
from jax_sph.case_setup import SimulationSetup
from omegaconf import DictConfig


class RLX(SimulationSetup):
    """Relaxation setup."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # relaxation configurations
        if self.case.mode == "rlx":
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0
        self.L = 2 * np.pi

    def _box_size2D(self, n_walls):
        return self.L * np.ones(2)

    def _box_size3D(self, n_walls):
        return self.L * np.ones(3)

    def _init_walls_2d(self):
        pass

    def _init_walls_3d(self):
        pass

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        return jnp.zeros_like(r)

    def _boundary_conditions_fn(self, state):
        return state
