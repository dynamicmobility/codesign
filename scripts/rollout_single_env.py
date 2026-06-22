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
from matplotlib import pyplot as plt


CONFIG_PATH = 'config/codesign_cheetah.yaml'
train_config = mop.utils.read_config(CONFIG_PATH)
env_params = mm.utils.config.create_config_dict(train_config['env_config'])

env = CodesignCheetah.generate_model(env_params, 'np', d=1.0)
base_name = train_config['name']
base_dir = train_config['save_dir'] 

n_objs    = mop.learning.inference.get_num_objectives(train_config)
# tradeoff  = np.random.dirichlet(alpha=np.ones(n_objs))
tradeoff = jnp.array([0.2, 0.8, 0.0])

# Train new policy
inference_fn = mop.learning.inference.load_mo_policy(
    config          = train_config,
    tradeoff        = tradeoff,
    deterministic   = True
)
frames, reward_plotter, _, _ = mm.eval.rollout_policy(
    inference_fn    = jax.jit(inference_fn),
    env             = env,
    T               = 4,
    width           = 640,
    height          = 480,
    camera          = 'track',
)
mm.utils.plotting.save_video(
frames,
env.dt,
Path(f'output/videos/{train_config["env"]}-rollout.mp4')
)

fig, axs = reward_plotter.plot()
plt.show()