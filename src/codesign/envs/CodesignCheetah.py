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

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):
        rewards = {
            'alive'  : self.reward_alive(),
            'energy' : self.reward_power(data, info),
            'height' : self.reward_height(data),
            'run'    : self.reward_run(info),
            'done'   : self.reward_done(done)
        }
        return rewards

    def reward_cot(self, data, info):
        m = 10.0 # mass of cheetah (this needs to be calculated from the spec)
        v = (info['xposafter'] - info['xposbefore']) / self.dt
        P = jnp.sum(data.qfrc_actuator[3:] * data.qvel[3:]) # power = force * velocity
        cot = P/(m*9.8*self._np.abs(v) + 1.0e-8)
        # make cot not inf when velocity is 0
        # cot = self._np.where(v > 0, cot, 1.0e8)
        return -cot

    def reward_power(self, data, info):
        P = jnp.sum(jnp.square(data.qfrc_actuator[3:])) # power = force * velocity
        return jnp.exp(-P/self.params.reward.sigmas.energy)

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