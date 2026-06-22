import jax

from codesign.envs.MAICheetah import MAICheetah
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
# envs = [MAICheetah.generate_model(env_params, 'np', d) for d in ds]
model = MAICheetah.generate_model(0.5)
env = MAICheetah(
    env_params=env_params,
    backend='np'
)

state = env.reset(None, model)
state = env.step(state, np.zeros(model.nu), model)
print('Done!')