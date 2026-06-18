"""Cheetah environment."""

from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from codesign.envs import CodesignInterface
from moplayground.envs.dmcontrol.interface import CheetahInterface
from moplayground.envs.dmcontrol.cheetah import MOCheetah
from mujoco_playground._src.dm_control_suite import common
import mujoco as mj
import jax.numpy as jnp
import numpy as np
from pathlib import Path

INTERFACE_PATH = Path(__file__).resolve().parent
class CodesignCheetah(MOCheetah):
    def __init__(self, env_params, backend, new_model):
        super().__init__(env_params, backend, xml_path=INTERFACE_PATH / "cheetah.xml")
        self._mj_model = new_model
        self.setup_swappable_backend(backend) # Once model is updated, data_init_fn also needs to be updated

    @classmethod
    def default_spec(cls) -> mj.MjSpec:
        return mj.MjSpec.from_file(
            filename=Path(INTERFACE_PATH / "cheetah.xml").as_posix(), 
            assets=common.get_assets()
        )

    # TODO: Override this with omsething that implements cost of transport    
    def reward_energy(self, action):
        return 4.0 - 1.0 * self._np.square(action).sum()

def generate_model(env_params, backend, d):
    shin_pos = jnp.array([0.2, 0, -0.26])
    new_shin_pos = shin_pos*d
    midpoint = new_shin_pos/2

    spec = CodesignCheetah.default_spec()

    # load spec from file
    spec.body("bshin").pos = new_shin_pos
    thigh_geom = spec.geom("bthigh")
    thigh_geom.pos = midpoint  # Change to desired position (x, y, z)
    thigh_geom.size[1] = jnp.linalg.norm(new_shin_pos)/2
    return CodesignCheetah(env_params, backend, spec.compile())

def resample_design(rng):
    length = np.random.uniform(shape=(1,), minval=0.5, maxval=2)
    return length