from src.codesign.envs import CodesignCheetah
import moplayground as mop
import minimal_mjx as mm
import mujoco
import argparse
import jax.numpy as jnp
from moplayground.envs.dmcontrol.cheetah import MOCheetah


CONFIG_PATH = 'envs/codesign_cheetah.yaml'
train_config = mop.utils.read_config(CONFIG_PATH)
env_params = mm.utils.config.create_config_dict(train_config['env_config'])
env = CodesignCheetah(env_params=env_params, backend="np")

