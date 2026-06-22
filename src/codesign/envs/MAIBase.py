"""Model-as-input (MAI) Cheetah environment."""

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
class MAIBase(SwappableBase):
    def __init__(
        self, 
        base_xml_path,
        env_params, 
        backend, 
        num_free
    ):
        SwappableBase.__init__(
            self,
            xml_path    = base_xml_path,
            env_params  = env_params,
            backend     = backend,
            num_free    = num_free
        )

    @classmethod
    @abstractmethod
    def generate_model(cls, d):
        pass

    def setup_swappable_backend(self, backend: str):
        """Sets up the backend for the environment."""
        super().setup_swappable_backend(backend)
        
        if backend == 'jnp':
            # Setup JAX backend
            self._mjx_model = mjx.put_model(self._mj_model)

            self._step_fn = lambda data, ctrl, model: mjx_env.step(
                model, data, ctrl, self.n_substeps
            )

            def mjx_data_init_fn(model, qpos, qvel, ctrl, time, xfrc_applied):
                data = mjx_env.init(
                    model, qpos=qpos, qvel=qvel, ctrl=ctrl
                ).replace(time=time, xfrc_applied=xfrc_applied)
                data = mjx.forward(model, data)
                return data.replace(ctrl=ctrl)

            self._data_init_fn = mjx_data_init_fn

        elif backend == 'np':
            # Setup numpy backend
            def init_data(model, time, qpos, qvel, ctrl, xfrc_applied):
                data = mj.MjData(model)
                data.time = time
                data.qpos = qpos
                data.qvel = qvel
                data.ctrl = ctrl
                data.xfrc_applied = xfrc_applied
                mj.mj_forward(model, data)
                data.ctrl = ctrl

                return data
            self._data_init_fn = lambda model, time, qpos, qvel, ctrl, xfrc_applied: init_data(
                model, time, qpos, qvel, ctrl, xfrc_applied
            )

            def mj_step(model, old_data, ctrl, n_substeps):
                new_data = mj.MjData(model)

                # Allocate state buffer
                state_size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
                state = np.empty(state_size)

                # Copy state from old_data
                mj.mj_getState(model, old_data, state, mj.mjtState.mjSTATE_FULLPHYSICS)

                # Set state in new_data
                mj.mj_setState(model, new_data, state, mj.mjtState.mjSTATE_FULLPHYSICS)

                # Step simulation
                for _ in range(n_substeps):
                    new_data.ctrl[:] = ctrl
                    mj.mj_step(model, new_data)

                return new_data
            self._step_fn = lambda data, ctrl, model, n_substeps=self.n_substeps: mj_step(
                model, data, ctrl, n_substeps
            )

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # @property
    # def mjx_model(self):
    #     raise AttributeError('Model-as-input Cheetah has no MJX model attribute. It must be passed as input.')
    
    # @property
    # def mj_model(self):
    #     raise AttributeError('Model-as-input Cheetah has no MJ model attribute. It must be passed as input.')

    @classmethod
    def default_spec(cls, xml_path: Path) -> mj.MjSpec:
        return mj.MjSpec.from_file(
            filename=xml_path.as_posix(),
            assets=common.get_assets()
        )


class MAIMO2SO:
    """Wrap a model-as-input multi-objective env to expose a scalar reward.

    The model-as-input analogue of
    ``moplayground.envs.generic.mobase.Multi2SingleObjective``: it replaces the wrapped
    env's vector reward with the inner product ``reward · weighting`` so a multi-objective
    env can be driven by single-objective machinery (PPO, plain rollouts). Unlike that
    class, the wrapped env's ``reset``/``step`` take the compiled ``mjx.Model`` as an
    explicit argument (``reset(rng, model)`` / ``step(state, action, model)``), so this
    wrapper forwards it through. All other attributes/methods are delegated to the
    underlying ``env`` via ``__getattr__``.

    Args:
        env: A model-as-input env (e.g. ``MAICheetah``) whose ``reset``/``step`` return
            states with vector rewards.
        weighting: Per-objective weights; length must match the env's reward dimension.
    """

    def __init__(self, env, weighting):
        self.env = env
        self.weighting = env._np.asarray(weighting)

    def _scalarize(self, state):
        return state.replace(
            reward=self.env._np.sum(state.reward * self.weighting)
        )

    def reset(self, rng: jax.Array, model) -> Any:
        return self._scalarize(self.env.reset(rng, model))

    def step(self, state, action: jax.Array, model) -> Any:
        return self._scalarize(self.env.step(state, action, model))

    def __getattr__(self, name):
        """Delegate any attribute not defined on the wrapper to the wrapped env."""
        return getattr(self.env, name)