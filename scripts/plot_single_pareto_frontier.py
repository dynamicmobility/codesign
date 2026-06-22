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
from moplayground.learning.inference import load_mo_policy
from moplayground.eval.pareto import get_morlax_fronts
from moplayground.utils.pareto import get_nondominated
from moplayground.utils.plotting import plot_pareto
from pathlib import Path
from matplotlib import pyplot as plt



CONFIG_PATH = 'config/codesign_cheetah.yaml'
train_config = mop.utils.read_config(CONFIG_PATH)
env_params = mm.utils.config.create_config_dict(train_config['env_config'])
ds = np.linspace(0.5, 2, 5)
base_name = train_config['name']
base_dir = train_config['save_dir'] 
env = CodesignCheetah.generate_model(env_params, 'jnp', d=1.0)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

rewards_over_iters, tradeoffs_over_iters = get_morlax_fronts(
    config          = train_config,
    rng             = jax.random.PRNGKey(21),
    env             = env,
    N_STEPS         = 500,
    NUM_ENVS        = 1024,
    save_results    = True,
)

print(rewards_over_iters.shape)
nd_idx = get_nondominated(rewards_over_iters[-1], epsilon=10)

ax = plot_pareto(
    ax          = ax,
    pareto      = rewards_over_iters[-1],
    directive   = tradeoffs_over_iters[-1],
    objective   = train_config.env_config.reward.optimization.labels,
    nondominated= nd_idx
)


plt.show()