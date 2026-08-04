"""Cheetah environment."""

from typing import Any

import jax
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from codesign.envs.CodesignBase import MOCodesignBase
from moplayground.envs.dmcontrol.interface import CheetahInterface
from moplayground.envs.dmcontrol.cheetah import MOCheetah
from mujoco.mjx._src.types import Model
import mujoco as mj
import jax.numpy as jnp
import numpy as np
from pathlib import Path

INTERFACE_PATH = Path(__file__).resolve().parent
    
class MOCodesignCheetah(MOCodesignBase):
    """Multi-Objective Cheetah Environment. 
    Objectives are speed, energy, and jumping height."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        mo_backend: MOCheetah = MOCheetah(
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
            self._np.array(CheetahInterface.DEFAULT_FF),
            self._np.array(CheetahInterface.DEFAULT_JT)
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
            'run'    : self.mo_backend.reward_run(info),
            'done'   : self.mo_backend.reward_done(done)
        }
        return rewards

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
            info['height'] < -0.35
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

    def default_design(self):
        return self._np.array([1.0])

    @property
    def design_limits(self):
        return self._np.array([[0.5], [2.0]])
    
    @classmethod
    def generate_model(cls, d):
        spec = MOCodesignCheetah.default_spec()
        
        geom_body_pairs = [
            ('fthigh', 'fshin'),
            ('fshin', 'ffoot'),
            ('ffoot', 'ftoe'),
            ('bthigh', 'bshin'),
            ('bshin', 'bfoot'),
            ('bfoot', 'btoe')
        ]
        for pair, d_dim in zip(geom_body_pairs, d):
            parent_geom, child_body = pair
            spec = cls.change_link_length(
                spec,
                parent_geom_name = parent_geom,
                child_body_name  = child_body,
                scale_factor     = d_dim
            )
        
        return spec.compile()

    @classmethod
    def _generate_model(cls, d):
        shin_pos = jnp.array([0.2, 0, -0.26])
        new_shin_pos = shin_pos*d
        midpoint = new_shin_pos/2

        spec = MOCodesignCheetah.default_spec()

        # load spec from file
        spec.body("bshin").pos = new_shin_pos
        thigh_geom = spec.geom("bthigh")
        thigh_geom.pos = midpoint  # Change to desired position (x, y, z)
        thigh_geom.size[1] = jnp.linalg.norm(new_shin_pos) / 2
        return spec.compile()