"""Codesign Cheetah environment."""

from abc import abstractmethod
from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from minimal_mjx.envs import SwappableBase
from mujoco_playground._src.dm_control_suite import common
import mujoco as mj
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from minimal_mjx.utils import EnvState

INTERFACE_PATH = Path(__file__).resolve().parent
class CodesignBase(SwappableBase):
    def __init__(
        self, 
        xml_path,
        env_params, 
        backend, 
        num_free
    ):
        SwappableBase.__init__(
            self,
            xml_path    = xml_path,
            env_params  = env_params,
            backend     = backend,
            num_free    = num_free
        )

    @classmethod
    @abstractmethod
    def generate_model(cls, d):
        pass
        
    @classmethod
    def default_spec(cls, xml_path: Path) -> mj.MjSpec:
        return mj.MjSpec.from_file(
            filename=xml_path.as_posix(),
            assets=common.get_assets()
        )
        
    @property
    def observation_size(self):
        """Observation structure, inferred from the env's nominal compiled model.
        """
        abstract_state = jax.eval_shape(
            lambda rng: self.reset(rng, self._mjx_model), jax.random.PRNGKey(0)
        )
        obs = abstract_state.obs
        if isinstance(obs, dict):
            return jax.tree_util.tree_map(lambda x: x.shape, obs)
        return obs.shape[-1]


class CodesignMO2SO:
    """Wrap a model-as-input multi-objective env to expose a scalar reward.

    Args:
        env: A model-as-input env (e.g. ``CodesignCheetah``) whose ``reset``/``step`` return
            states with vector rewards.
        weighting: Per-objective weights; length must match the env's reward dimension.
    """

    def __init__(self, env: CodesignBase, weighting):
        self.env = env
        self.weighting = env._np.asarray(weighting)

    def _scalarize(self, state):
        return state.replace(
            reward=self.env._np.sum(state.reward * self.weighting)
        )

    def reset(self, rng: jax.Array) -> Any:
        return self._scalarize(self.env.reset(rng, self.env.mjx_model if self.env.backend == 'jnp' else self.env.mj_model))

    def step(self, state, action: jax.Array) -> Any:
        return self._scalarize(self.env.step(state, action, self.env.mjx_model if self.env.backend == 'jnp' else self.env.mj_model))

    def __getattr__(self, name):
        """Delegate any attribute not defined on the wrapper to the wrapped env."""
        return getattr(self.env, name)