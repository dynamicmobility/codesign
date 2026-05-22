from codesign.envs.CodesignCheetah import CodesignCheetah
import moplayground as mop
from codesign.eval import rollout_policy
import minimal_mjx as mm
import mujoco
import argparse
import jax.numpy as jnp
from moplayground.envs.dmcontrol.cheetah import MOCheetah
from pathlib import Path


CONFIG_PATH = 'config/codesign_cheetah.yaml'
train_config = mop.utils.read_config(CONFIG_PATH)
env_params = mm.utils.config.create_config_dict(train_config['env_config'])
env = CodesignCheetah(env_params=env_params, backend="np")

# env.set_mj_model(env.generate_model(1))

def do_nothing(obs, rng):
    return jnp.zeros(env.action_size), 0.0

frames, reward_plotter, _, _ = rollout_policy(
        inference_fn    = do_nothing,
        env             = env,
        T               = 2,
        width           = 640,
        height          = 480,
        camera          = 'track',
    )

mm.utils.plotting.save_video(
    frames,
    env.dt,
    Path(f'output/videos/{train_config['env']}-rollout.mp4')
)