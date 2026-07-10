"""Cheetah environment."""

from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from codesign.envs.CodesignBase import CodesignBase
from moplayground.envs.dmcontrol.interface import CheetahInterface
from moplayground.envs.dmcontrol.cheetah import MOCheetah
from mujoco.mjx._src.types import Model
import mujoco as mj
import jax.numpy as jnp
import numpy as np
from pathlib import Path

INTERFACE_PATH = Path(__file__).resolve().parent
    
class RHex(CodesignBase):
    """Multi-Objective Cheetah Environment. 
    Objectives are speed, energy, and jumping height."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
    ):
        CodesignBase.__init__(
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

        site_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, "bfoot_tip")
        probe = self._data_init_fn(
            model        = model,
            qpos         = qpos,
            qvel         = qvel,
            ctrl         = ctrl,
            time         = 0.0,
            xfrc_applied = self._np.zeros((model.nbody, 6)),
        )
        ground_margin = 0.02
        tip_z = probe.site_xpos[site_id][2]
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
            'run'    : self.reward_run(info),
        }
        return rewards

    def reward_power(self, data, info):
        P = jnp.sum(jnp.square(data.qfrc_actuator[6:])) # power = force * velocity
        assert len(data.qfrc_actuator[6:]) == 6
        return -P
    
    def reward_run(self, info):
        return info['xposafter'] - info['xposbefore']

    
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
        leg_thickness = 0.0016
        leg_radius = 0.032
        stance = 0.0

        d_default = np.array([leg_thickness, leg_radius, stance])
        return cls.generate_model(d_default)
    

    @classmethod
    def generate_model(cls, d):

        leg_width = 0.014
        leg_thickness = d[0]
        mid_radius = d[1]
        stance = d[2]
        n_segments = 8
        spec = mj.MjSpec()

        front_radius = mid_radius + stance
        back_radius = mid_radius - stance


        spec.option.integrator = mj.mjtIntegrator.mjINT_EULER
        spec.option.solver = mj.mjtSolver.mjSOL_NEWTON
        spec.option.cone = mj.mjtCone.mjCONE_PYRAMIDAL
        spec.option.timestep = 0.0001
        spec.option.iterations = 4
        # ls_iterations controls the number of line-search iterations for MPRGP/solver
        spec.option.ls_iterations = 8
        # # Disable euler implicit damping (match eulerdamp="disable")
        # spec.option.flags.eulerdamp = False
        

        # Contact parameters used by multiple geoms
        contact_params = dict(
            condim=3,
            friction=(0.4, 0.1, 0.1),
            solref = [0.005, 5],
            solimp = [0.0, 0.95, 0.0001, 0.1, 2]
        )

        # Add ground plane
        spec.worldbody.add_geom(
            name="ground",
            type=mj.mjtGeom.mjGEOM_PLANE,
            size=(1.0, 1.0, 0.1),
            pos=(0.0, 0.0, -0.05),
            contype = 1,
            conaffinity = 1,
            **contact_params,
        )
        
        body = spec.worldbody.add_body(name="robot_body", pos=(0.0, 0.0, 0.0))

        # Free joint connecting the robot body to the world
        body.add_joint(
            name="root",
            type=mj.mjtJoint.mjJNT_FREE,
        )

        body.add_geom(
            name="robot_body",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=(0.093, 0.05, 0.018),
            pos=(0.0, 0.0, 0.0),
            mass=0.4,
            **contact_params,
        )


        DENSITY = 1.23 # g/cm^3
        YOUNGS_MODULUS = 3.2e9 # Pa
        leg_moment_of_area = (leg_width*leg_thickness**3)/12.0 # m^4
        # Add legs
        def add_leg(leg_name, base_pos, radius):
            arc_length = np.pi * radius
            seg_len = arc_length / n_segments
            dtheta = np.pi / n_segments
            K_effective = 2*YOUNGS_MODULUS * leg_moment_of_area / (np.pi * radius**3) # N/m
            print(K_effective)
            # Compute jacobian
            J2 = jnp.array([seg_len*jnp.cos(dtheta*i) for i in range(n_segments)])
            Kt = K_effective*jnp.dot(J2,J2)

            # Compute K

            # Base hinge at body with an actuator.
            parent = body
            next_geom_pos = (seg_len/2, 0.0, 0.0)

            for i in range(n_segments):
                segment_mass = DENSITY * leg_thickness * leg_width * seg_len * 1000.0 # g/cm^3 -> kg/m^3
                this_joint_name = f"{leg_name}_joint_{i}"
                this_body = parent.add_body(
                    name=f"{leg_name}_seg_{i}",
                    pos=tuple(base_pos) if i == 0 else (seg_len, 0., 0.),
                    euler = tuple((0.0, 0.0, 0.0)) if i == 0 else (0.0, np.rad2deg(dtheta), 0.0),
                )
                this_body.add_geom(
                    name=f"{leg_name}_geom_{i}",
                    type=mj.mjtGeom.mjGEOM_BOX,
                    size=(seg_len * 0.5, leg_width * 0.5, leg_thickness * 0.5),
                    pos=next_geom_pos,
                    mass=segment_mass,
                    **contact_params,
                    contype = 2,
                    conaffinity = 1,
                    # euler = tuple(0.0, 0.0, 0.0) if i == 0 else (0.0, dtheta, 0.0),
                )
                if(i == 0):
                    this_body.add_joint(
                        name=this_joint_name,
                        type=mj.mjtJoint.mjJNT_HINGE,
                        axis=(0.0, 1.0, 0.0),
                        stiffness=0.0,
                        damping=1.0,
                    )
                    spec.add_actuator(
                        name=f"{leg_name}_act",
                        target=this_joint_name,
                        trntype=mj.mjtTrn.mjTRN_JOINT,
                        biastype=mj.mjtBias.mjBIAS_NONE,
                    )
                else:
                    this_body.add_joint(
                        name=this_joint_name,
                        type=mj.mjtJoint.mjJNT_HINGE,
                        axis=(0.0, 1.0, 0.0),
                        stiffness=Kt,
                        damping=0.01,
                    )

                parent = this_body
            
        add_leg("leg_bl", np.array([-0.08, 0.064, 0.0]), radius=back_radius)
        add_leg("leg_ml", np.array([0.0, 0.08, 0.0]), radius=mid_radius)
        add_leg("leg_fl", np.array([0.08, 0.064, 0.0]), radius=front_radius)

        add_leg("leg_br", np.array([-0.08, -0.064, 0.0]), radius=back_radius)
        add_leg("leg_mr", np.array([0.0, -0.08, 0.0]), radius=mid_radius)
        add_leg("leg_fr", np.array([0.08, -0.064, 0.0]), radius=front_radius)
        return spec.compile()

# def resample_design(rng):
#     length = np.random.uniform(shape=(1,), minval=0.5, maxval=2)
#     return length