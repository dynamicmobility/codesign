"""Cheetah environment."""

from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from codesign.envs import CodesignInterface
from codesign.envs.MAIBase import MAIBase
from moplayground.envs.dmcontrol.interface import CheetahInterface
from moplayground.envs.dmcontrol.cheetah import MOCheetah
from mujoco.mjx._src.types import Model
import mujoco as mj
import jax.numpy as jnp
import numpy as np
from pathlib import Path

INTERFACE_PATH = Path(__file__).resolve().parent
    
class MAICheetah(MAIBase, MOCheetah):
    """Multi-Objective Cheetah Environment. 
    Objectives are speed, energy, and jumping height."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        MAIBase.__init__(
            self,
            base_xml_path     = INTERFACE_PATH / "cheetah.xml",
            env_params        = env_params,
            backend           = backend,
            num_free          = 3
        )

        MOCheetah.__init__(
            self,
            env_params    = env_params,
            backend       = backend,
            xml_path      = INTERFACE_PATH / 'cheetah.xml'
        )

    def reset(self, rng: jax.Array, model: Model) -> mjx_env.State:
        # input better initialization parameters as a func of mjx_model here
        qpos = self._np.hstack([
            self._np.array(CheetahInterface.DEFAULT_FF),
            self._np.array(CheetahInterface.DEFAULT_JT)
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = self._np.zeros(self.mj_model.nu)
        
        data = self._data_init_fn(
            model        = model,
            qpos         = qpos,
            qvel         = qvel,
            ctrl         = ctrl,
            time         = 0.0,
            xfrc_applied = self._np.zeros((model.nbody, 6)),
        )
        parent_state = MAIBase.reset(
            self,
            rng            = rng,
            data           = data,
            history_length = self.params.history_length
        )
        info = {}
        info['xposbefore'] = 0.0
        info['xposafter']  = 0.01
        info['ang']        = 0.0
        info = parent_state.info | info

        done = self._np.array(0.0)
        rewards = self.reward_function(
            data   = data,
            action = ctrl,
            info   = info,
            done   = False,
        )
        reward, metrics = self.get_reward_and_metrics(rewards, {})
        
        obs = self._get_obs(data, parent_state.info)
        return self._state_init_fn(data, obs, reward, done, metrics, info)
    
    def state_vector(self, data):
        return self._np.hstack([data.qpos, data.qvel])

    def step(self, state: mjx_env.State, action: jax.Array, model: Model) -> mjx_env.State:
        state.info['xposbefore'] = state.data.qpos[0]
        action = self._np.clip(
            self.params.action_scale * action, 
            -1.0,
            1.0
        )
        data = self._step_fn(state.data, action, model)
        state.info['xposafter'] = data.qpos[0]
        state.info['ang']       = data.qpos[2]
        
        done = self.fall_termination(state.info)
        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )
        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
        obs = self._get_obs(
            data,
            state.info
        )
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)
    
    @property
    def action_size(self):
        return 6

    @property
    def observation_size(self):
        """Observation structure, inferred with a nominal design.

        The base ``MjxEnv.observation_size`` traces ``self.reset(rng)``, but this env is
        model-as-input (``reset(rng, model)``), so we trace against a model built from a
        nominal design (d=1.0). Returns a dict of shapes (obs is a dict).
        """
        model = mjx.put_model(self.generate_model(1.0))
        abstract_state = jax.eval_shape(
            lambda rng: self.reset(rng, model), jax.random.PRNGKey(0)
        )
        obs = abstract_state.obs
        if isinstance(obs, dict):
            return jax.tree_util.tree_map(lambda x: x.shape, obs)
        return obs.shape[-1]

    @classmethod
    def default_spec(cls) -> mj.MjSpec:
        return MAIBase.default_spec(xml_path=INTERFACE_PATH / "cheetah.xml")
    

    @classmethod
    def generate_model(cls, d):
        shin_pos = jnp.array([0.2, 0, -0.26])
        new_shin_pos = shin_pos*d
        midpoint = new_shin_pos/2

        spec = MAICheetah.default_spec()

        # load spec from file
        spec.body("bshin").pos = new_shin_pos
        thigh_geom = spec.geom("bthigh")
        thigh_geom.pos = midpoint  # Change to desired position (x, y, z)
        thigh_geom.size[1] = jnp.linalg.norm(new_shin_pos)/2
        return spec.compile()

# def resample_design(rng):
#     length = np.random.uniform(shape=(1,), minval=0.5, maxval=2)
#     return length