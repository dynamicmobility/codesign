import jax

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
base_name = train_config['name']

for d, env in zip(ds, envs):
    train_config['name'] = base_name + f'-d{d}'
    name = train_config['save_dir'] + '/' + base_name + f'-d{d}'
    run = mm.utils.logging.initialize_wandb(
        name    = name.replace('/', ''),
        entity  = 'vmadabushi3-georgia-institute-of-technology',
        project = 'codesign-cheetah'
    )
    make_inference_fn, params = mop.learning.train_policy(train_config, env, env, run)
    n_objs    = mop.learning.inference.get_num_objectives(train_config)
    tradeoff  = np.random.dirichlet(alpha=np.ones(n_objs))
    inference_fn = make_inference_fn(
            params        = params,
            deterministic = True,
            directive     = tradeoff,
            single_policy = True
        )
    frames, reward_plotter, _, _ = mm.eval.rollout_policy(
            inference_fn    = jax.jit(inference_fn),
            env             = env,
            T               = 2,
            width           = 640,
            height          = 480,
            camera          = 'track',
        )
    mm.utils.plotting.save_video(
        frames,
        env.dt,
        Path(f'output/videos/{train_config["env"]}-{d}-rollout.mp4')
    )