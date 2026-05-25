from codesign.envs.CodesignCheetah import CodesignCheetah
from codesign.envs import CodesignCheetah
import moplayground as mop
import minimal_mjx as mm
import mujoco
import argparse
import jax.numpy as jnp
import numpy as np
from moplayground.envs.dmcontrol.cheetah import MOCheetah
from pathlib import Path


CONFIG_PATH = 'config/codesign_cheetah.yaml'
train_config = mop.utils.read_config(CONFIG_PATH)
env_params = mm.utils.config.create_config_dict(train_config['env_config'])
ds = np.linspace(0.5, 2, 5)
envs = [CodesignCheetah.generate_model(env_params, 'np', d) for d in ds]

for d, env in zip(ds, envs):
    def do_nothing(obs, rng):
        return jnp.zeros(env.action_size), 0.0

    frames, reward_plotter, _, _ = mm.eval.rollout_policy(
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
        Path(f'output/videos/{train_config['env']}-{d}-rollout.mp4')
    )