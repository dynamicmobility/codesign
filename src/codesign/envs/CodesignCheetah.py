"""Cheetah environment."""

from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from src.codesign.envs import CodesignInterface
from moplayground.envs.dmcontrol.interface import CheetahInterface
from moplayground.envs.dmcontrol.cheetah import MOCheetah
import jax.numpy as jnp
from pathlib import Path

class CodesignCheetah(MOCheetah, CodesignInterface.CodesignInterface):
    # Implements a swappable leg length for MOCheetah
    # CodesignInterface forces modify_design and resample_design to be implemented

    def __init__(self, env_params, backend):
        super().__init__(env_params, backend)

    def generate_model(self, d):
        shin_pos = jnp.array([0.2, 0, -0.26])
        new_shin_pos = shin_pos*d
        midpoint = new_shin_pos/2

        # load spec from file
        spec = self._spec

        spec.body("bshin").pos = new_shin_pos
        thigh_geom = spec.geom("bthigh")
        thigh_geom.pos = midpoint  # Change to desired position (x, y, z)
        thigh_geom.size[1] = jnp.linalg.norm(new_shin_pos)/2
        return spec.compile() # This may need to be swappable backended to return either a mujoco model or mjx model

    def resample_design(self, rng):
        length = self._uniform(rng, shape=(1,), minval=0.5, maxval=2)
        return length

    # Randomizes leg length upon reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, leg_length_key  = self._split(rng, 2)

        # Checks params dict to see if design should be resampled, and if so, how
        codesign = self.params.codesign
        if(codesign.enabled):
            if(codesign.strategy == 'random'):
                d = self.resample_design(leg_length_key)
            elif(codesign.strategy == 'predictor'):
                # This is where the design predictor would go
                raise NotImplementedError()
        else:
            d = 1

        new_model = self.generate_model(d)
        parent_state = super().reset()
        info = {}
        info['design'] = d
        info['model'] = new_model
        parent_state.info = parent_state.info | info # Can you do this?

        return parent_state

    def step(self, state, action):
        super().step(state, action)