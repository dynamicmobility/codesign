"""Cheetah environment."""

from typing import Any

import jax
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from codesign.envs.codesign_base import MOCodesignBase
import moplayground as mop
from mujoco.mjx._src.types import Model
import mujoco as mj
import jax.numpy as jnp
import numpy as np
from pathlib import Path

INTERFACE_PATH = Path(__file__).resolve().parent
    
class MOCodesignCheetah(MOCodesignBase):
    """Multi-Objective Cheetah Environment. 
    Objectives are speed, energy, and jumping height."""

    GEOM_BODY_PAIRS = [
        ('fthigh', 'fshin'),
        ('fshin', 'ffoot'),
        ('ffoot', 'ftoe'),
        ('bthigh', 'bshin'),
        ('bshin', 'bfoot'),
        ('bfoot', 'btoe')
    ]

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        mo_backend: mop.MOCheetah = mop.MOCheetah(
            env_params = env_params,
            backend = backend,
            xml_path = INTERFACE_PATH / "xmls" / "cheetah.xml",
        )
        
        super().__init__(
            xml_path          = INTERFACE_PATH / "xmls" / "cheetah.xml",
            env_params        = env_params,
            backend           = backend,
            num_free          = 3,
            mo_backend        = mo_backend
        )

    def reset(self, rng: jax.Array, model: Model) -> mjx_env.State:
        # input better initialization parameters as a func of mjx_model here
        qpos = self._np.hstack([
            self._np.array(mop.CheetahInterface.DEFAULT_FF),
            self._np.array(mop.CheetahInterface.DEFAULT_JT)
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = self._np.zeros(self.mj_model.nu)

        bsite_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, "bfoot_tip")
        fsite_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, "ffoot_tip")
        probe = self._data_init_fn(
            model        = model,
            qpos         = qpos,
            qvel         = qvel,
            ctrl         = ctrl,
            time         = 0.0,
            xfrc_applied = self._np.zeros((model.nbody, 6)),
        )
        ground_margin = 0.02
        btip_z = probe.site_xpos[bsite_id][2]
        ftip_z = probe.site_xpos[fsite_id][2]
        tip_z = self._np.min(self._np.asarray([btip_z, ftip_z]))
        qpos = self._set_val_fn(qpos, qpos[1] - tip_z + ground_margin, 1, 2)
        
        data = self._data_init_fn(
            model        = model,
            qpos         = qpos,
            qvel         = qvel,
            ctrl         = ctrl,
            time         = 0.0,
            xfrc_applied = self._np.zeros((model.nbody, 6)),
        )
        info = {}
        info['xposbefore'] = 0.0
        info['xposafter']  = 0.01
        info['ang']        = data.qpos[2]
        info['height']     = data.qpos[1]

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
        state.info['xposbefore'] = state.data.qpos[0]
        action = self._np.clip(
            self.params.action_scale * action, 
            -1.0,
            1.0
        )
        data = self._step_fn(state.data, action, model)
        state.info['xposafter'] = data.qpos[0]
        state.info['ang']       = data.qpos[2]
        state.info['height']    = data.qpos[1]
        
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
            'height' : self.mo_backend.reward_height(data),
            'run'    : self.reward_run(info),
            'done'   : self.mo_backend.reward_done(done)
        }
        return rewards
    
    def reward_run(self, info):
        reward_run = (info['xposafter'] - info['xposbefore']) / self.dt
        return reward_run
    
    def reward_power(self, data, info):
        P = jnp.sum(jnp.square(data.qfrc_actuator[3:])) # power = force * velocity
        return -P 
    
    def fall_termination(
        self,  
        info: dict
    ):
        upside_down = self._np.array(
            ~(abs(info['ang']) < self._np.deg2rad(80))
        )

        too_low = self._np.array(
            info['height'] < -0.55
        )
        return upside_down | too_low

    def _get_obs(self, data, info):
        return self.mo_backend._get_obs(data, info)
    
    @property
    def action_size(self):
        return 6

    @classmethod
    def default_spec(cls) -> mj.MjSpec:
        return super().default_spec(xml_path=INTERFACE_PATH / "xmls" / "cheetah.xml")

    @property
    def design_limits(self):
        """``[low, high]``, one entry per ``GEOM_BODY_PAIRS`` link scale."""
        return self._np.array([
            self.params.codesign.low,
            self.params.codesign.high,
        ])
    def default_design(self):
        return self.params.codesign.default_design

    
    @classmethod
    def generate_model(cls, d, textures: bool = True):
        spec = cls.default_spec()
        if not textures:
            cls.shrink_textures(spec)
        
        for pair, d_dim in zip(cls.GEOM_BODY_PAIRS, d, strict=True):
            parent_geom, child_body = pair
            spec = cls.change_link_length(
                spec,
                parent_geom_name = parent_geom,
                child_body_name  = child_body,
                scale_factor     = d_dim
            )
        
        return spec.compile()
    

class MOCodesignCheetah1D(MOCodesignCheetah):

    GEOM_BODY_PAIRS = [
        ('bthigh', 'bshin'),
    ]

class MOCodesignCheetah1DOldEnergy(MOCodesignCheetah):

    GEOM_BODY_PAIRS = [
        ('bthigh', 'bshin'),
    ]

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):
        rewards = {
            'alive'  : self.reward_alive(),
            'energy' : self.mo_backend.reward_energy(action),
            'height' : self.mo_backend.reward_height(data),
            'run'    : self.reward_run(info),
            'done'   : self.mo_backend.reward_done(done)
        }
        return rewards

class MOCodesignCheetah2D(MOCodesignCheetah):

    GEOM_BODY_PAIRS = [
        ('bthigh', 'bshin'),
        ('fthigh', 'fshin'),
    ]

class MOCodesignCheetahBackLegs(MOCodesignCheetah):

    GEOM_BODY_PAIRS = [
        ('bthigh', 'bshin'),
        ('bshin', 'bfoot'),
        ('bfoot', 'btoe')
    ]

class MOCodesignCheetahFrontLegs(MOCodesignCheetah):

    GEOM_BODY_PAIRS = [
        ('fthigh', 'fshin'),
        ('fshin', 'ffoot'),
        ('ffoot', 'ftoe'),
    ]

class MOCodesignCheetah4D(MOCodesignCheetah):

    GEOM_BODY_PAIRS = [
        ('fthigh', 'fshin'),
        ('fshin', 'ffoot'),
        ('ffoot', 'ftoe'),
        ('bthigh', 'bshin')
    ]


class MOCodesignCheetah5D(MOCodesignCheetah):

    GEOM_BODY_PAIRS = [
        ('fthigh', 'fshin'),
        ('fshin', 'ffoot'),
        ('ffoot', 'ftoe'),
        ('bthigh', 'bshin'),
        ('bshin', 'bfoot'),
    ]
