"""Two Axis Positioning environment."""

from typing import Any

import jax
from ml_collections import config_dict
from moplayground.envs.generic.mobase import MultiObjectiveBase
from mujoco import mjx
from mujoco_playground._src import mjx_env

from codesign.envs.CodesignBase import CodesignBase
from mujoco.mjx._src.types import Model
import mujoco as mj
import jax.numpy as jnp
import numpy as np
from pathlib import Path

INTERFACE_PATH = Path(__file__).resolve().parent
xml_name = "twoaxis.xml"
    
class TwoAxis(MultiObjectiveBase, CodesignBase):
    """Multi-Objective Two-Axis Environment. 
    Objectives are x and y position tracking."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        MultiObjectiveBase.__init__(
            self,
            xml_path          = INTERFACE_PATH / "xmls" / xml_name,
            env_params        = env_params,
            backend           = backend,
            num_free          = 0
        )
        CodesignBase.__init__(
            self,
            xml_path     = INTERFACE_PATH / "xmls" / xml_name,
            env_params        = env_params,
            backend           = backend,
            num_free          = 0
        )

    def default_spec(cls):
        return CodesignBase.default_spec(xml_path=INTERFACE_PATH / "xmls" / xml_name)

    def reset(self, rng: jax.Array, model: Model) -> mjx_env.State:
        # input better initialization parameters as a func of mjx_model here
        qpos = self._np.zeros(self.mj_model.nq)
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
        info = {}

        done = self._np.array(0.0)
        rewards = self.reward_function(
            data   = data,
            action = ctrl,
            info   = info,
            done   = False,
        )
        reward, metrics = self.get_reward_and_metrics(rewards, {})
        
        obs = self._get_obs(data, info)
        return self._state_init_fn(data, obs, reward, done, metrics, info)
    
    def state_vector(self, data):
        return self._np.hstack([data.qpos, data.qvel])

    def step(self, state: mjx_env.State, action: jax.Array, model: Model) -> mjx_env.State:
        action = self._np.clip(
            self.params.action_scale * action, 
            -self._np.inf,
            self._np.inf
        )
        data = self._step_fn(state.data, action, model)
        
        done = self.termination(state.info)
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
    
    def _get_obs(self, data: Any, info: dict) -> jax.Array:
        """Returns the observation vector for the environment."""
        obs = self.state_vector(data)
        return obs

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):
        rewards = {
            'x_track' : self.x_track_reward(data, info),
            'y_track' : self.y_track_reward(data, info),
            'force'   : self.force_reward(data, action),
        }
        return rewards

    def x_track_reward(self, data, info):
        """Reward for tracking the x position of the target."""
        xpos = data.qpos[0]
        xvel = data.qvel[0]
        target_xpos = 2.0
        reward = -(xpos - target_xpos)**2 - 0.1*xvel**2
        return reward

    def y_track_reward(self, data, info):
        """Reward for tracking the y position of the target."""
        ypos = data.qpos[1]
        yvel = data.qvel[1]
        target_ypos = 2.0
        reward = -(ypos - target_ypos)**2 - 0.1*yvel**2
        return reward

    def force_reward(self, data, action):
        # return -data.actuator_force[0]**2 - data.actuator_force[1]**2
        return -self._np.linalg.norm(action)**2

    def termination( self,  info: dict):
        return self._np.array(False)

    
    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        """Observation structure, inferred with a nominal design.

        The base ``MjxEnv.observation_size`` traces ``self.reset(rng)``, but this env is
        model-as-input (``reset(rng, model)``), so we trace against a model built from a
        nominal design (d=1.0). Returns a dict of shapes (obs is a dict).
        """
        model = mjx.put_model(self.generate_model(0.5))
        abstract_state = jax.eval_shape(
            lambda rng: self.reset(rng, model), jax.random.PRNGKey(0)
        )
        obs = abstract_state.obs
        if isinstance(obs, dict):
            return jax.tree_util.tree_map(lambda x: x.shape, obs)
        return obs.shape[-1]

    @classmethod
    def default_spec(cls) -> mj.MjSpec:
        return CodesignBase.default_spec(xml_path=INTERFACE_PATH / "xmls" / xml_name)
    
    # d goes from 0 to 1 and modifies the ratio of x force range to y force range
    @classmethod
    def generate_model(cls, d):
        spec = cls.default_spec()
        d = np.clip(d, 0.01, 0.99)
        max_force = 10.0
        # load spec from file
        spec.actuator("rootx").forcerange = [-d*max_force, d*max_force]
        spec.actuator("rooty").forcerange = [-(1-d)*max_force, (1-d)*max_force]
        return spec.compile()

# def resample_design(rng):
#     length = np.random.uniform(shape=(1,), minval=0.5, maxval=2)
#     return length