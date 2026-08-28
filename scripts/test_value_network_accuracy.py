from dataclasses import dataclass
from typing import Optional

from codesign.utils import model as model_lib
from codesign.learning.inference import load_mo_design_value_hypernetwork
from functools import partial

import minimal_mjx as mm
import moplayground as mop
import codesign
import numpy as np
import jax.numpy as jnp
import jax
import time
from matplotlib import pyplot as plt


# CONFIG_PATH = "config/design_hypernetwork/cheetah6D.yaml"
# TODO: generate dataset for 95mhoslv
DATASET = 'scripts/outputs/datasets/95mhoslv/sweep_2d.npz'
CONFIG_PATH = "/home/varun/Documents/DynamoLab/Learning/codesign/results/wandb-downloads/95mhoslv/config.yaml"
config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
env, env_params = codesign.load_env(config=config, backend="jnp")
lower_bounds = np.array([env_params.codesign.low])
upper_bounds = np.array([env_params.codesign.high])
dim = len(lower_bounds)

usup  = codesign.DesignTradeoffDataset.load(DATASET) # Can try with a more complex dataset
names = codesign.utils.plotting.objective_labels(usup.objectives)
n_tradeoffs = usup.tradeoffs.shape[0]
n_designs = usup.designs.shape[0]
n_objectives = usup.rewards.shape[-1]

print(f"Dataset with {n_objectives} objectives, {n_designs} designs, {n_tradeoffs} tradeoffs")
designs_normalized = model_lib.normalize_design(usup.designs, lower_bounds, upper_bounds)
designs_tile = np.tile(designs_normalized, (n_tradeoffs, 1))
tradeoffs_tile = np.repeat(usup.tradeoffs, n_designs, axis=0)


_, reset = mm.get_step_reset(env)

print("Building models")
models = model_lib.build_batched_model(env, designs_tile)

# Load value inference
value_inference_fn, val_params = load_mo_design_value_hypernetwork(config)
print("Resetting")
state = jax.vmap(reset, in_axes=(None, 0))(0, models)

print(designs_normalized.shape)

print(designs_tile.shape)
print(tradeoffs_tile.shape)
# Load the trained, design-conditioned value network
value_fn = value_inference_fn(val_params, designs_tile, tradeoffs_tile)
obs = jax.vmap(env._get_obs)(state.data, state.info)
values_from_val_fn = np.sum(value_fn(obs) * tradeoffs_tile, axis=1)

total_rewards_rollout = np.sum(usup.rewards.reshape(-1, usup.rewards.shape[-1])*tradeoffs_tile, axis=1)

plt.title("Predicted vs Actual total rewards")
plt.scatter(values_from_val_fn, total_rewards_rollout)
plt.xlabel("Predicted reward from value function")
plt.ylabel("Measured reward from rollout")
plt.show()
