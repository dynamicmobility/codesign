"""Codesign Cheetah environment."""

from abc import abstractmethod
from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from minimal_mjx.envs import SwappableBase
import moplayground as mop
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
    @abstractmethod
    def default_design(self):
        """Returns a _np array of the default design parameters."""
        raise NotImplementedError()

    @property
    def observation_size(self):
        """Observation structure, inferred from the env's nominal compiled model.
        """
        abstract_state = jax.eval_shape(
            lambda rng: self.reset(rng, self._mj_model), jax.random.PRNGKey(0)
        )
        obs = abstract_state.obs
        if isinstance(obs, dict):
            return jax.tree_util.tree_map(lambda x: x.shape, obs)
        return obs.shape[-1]
    
    @property
    @abstractmethod
    def design_limits(self):
        """Returns a _np array [low, high], where low/high are the same shape as
        the design parameter d."""
        raise NotImplementedError()
    
    @classmethod
    def change_link_length(
        cls,
        spec: mj.MjSpec,
        parent_geom_name: str,
        child_body_name: str | None, 
        scale_factor
    ):
        """Parent geom must be a capsule element"""
        # spec.body(parent_body_name).pos     *= scale_factor
        spec.geom(parent_geom_name).pos     *= scale_factor
        spec.geom(parent_geom_name).size[1] *= scale_factor
        
        if child_body_name is not None:
            spec.body(child_body_name).pos      *= scale_factor
        return spec
        
        
        
        
    
    
class MOCodesignBase(CodesignBase):
    
    def __init__(
        self, 
        xml_path,
        env_params, 
        backend, 
        num_free,
        mo_backend=None
    ):
        super().__init__(
            xml_path    = xml_path,
            env_params  = env_params,
            backend     = backend,
            num_free    = num_free
        )
        
        if mo_backend is None:
            self.mo_backend = mop.envs.generic.MultiObjectiveBase(
                xml_path    = xml_path,
                env_params  = env_params,
                backend     = backend,
                num_free    = num_free
            )
        else:
            self.mo_backend: mop.envs.generic.MultiObjectiveBase = mo_backend
    
    def get_reward_and_metrics(self, rewards, metrics):
        return self.mo_backend.get_reward_and_metrics(rewards, metrics)
    
    @property
    def objectives(self):
        return self.mo_backend.objectives
    
    @property
    def shared_objectives(self):
        return self.mo_backend.shared_objectives


class CodesignMO2SO:
    """Wrap a model-as-input multi-objective env to expose a scalar reward.

    Args:
        env: A model-as-input env (e.g. ``CodesignCheetah``) whose ``reset``/``step`` return
            states with vector rewards.
        weighting: Per-objective weights; length must match the env's reward dimension.
    """

    def __init__(self, env: MOCodesignBase, weighting):
        self.env = env
        self.weighting = env._np.asarray(weighting)

    def _scalarize(self, state):
        return state.replace(
            reward=self.env._np.sum(state.reward * self.weighting)
        )

    def reset(self, rng: jax.Array, model = None) -> Any:
        if(model == None):
            return self._scalarize(self.env.reset(rng, self.env.mjx_model if self.env.backend is not 'np' else self.env.mj_model))
        return self._scalarize(self.env.reset(rng, model))

    def step(self, state, action: jax.Array, model = None) -> Any:
        if(model == None):
            return self._scalarize(self.env.step(state, action, self.env.mjx_model if self.env.backend is not 'np' else self.env.mj_model))
        return self._scalarize(self.env.step(state, action, model))

    def __getattr__(self, name):
        """Delegate any attribute not defined on the wrapper to the wrapped env."""
        return getattr(self.env, name)
    
class Codesign2SingleDesign:
    """Wrap a codesign (mulit or single objective) env to use a preset model.

    Args:
        env: A codesign env (e.g. ``CodesignCheetah``).
        design: design vector to set as the single design.
    """

    def __init__(self, env: CodesignBase, design):
        self.env = env
        
        model = env.generate_model(np.asarray(design))
        self.model = model # if env.backend == 'np' else mjx.put_model(model, impl=env.backend)

    def reset(self, rng: jax.Array) -> Any:
        return self.env.reset(rng, self.model)

    def step(self, state, action: jax.Array) -> Any:
        return self.env.step(state, action, self.model)

    def __getattr__(self, name):
        """Delegate any attribute not defined on the wrapper to the wrapped env."""
        return getattr(self.env, name)