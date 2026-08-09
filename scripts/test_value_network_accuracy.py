from dataclasses import dataclass
from typing import Optional

from codesign.utils import model as model_lib
from codesign.eval.parallel_eval import rollout_so_parallel
from minimal_mjx.eval import policy as policy_lib
from codesign.learning.inference import load_design_hypernetwork, load_design_value_hypernetwork
from functools import partial

import minimal_mjx as mm
import moplayground as mop
import codesign
import numpy as np
import jax.numpy as jnp
import jax
import time
from matplotlib import pyplot as plt


CONFIG_PATH = "config/design_hypernetwork/cheetah6D.yaml"
config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
env, env_params = codesign.load_env(config=config, backend="jnp")
lower_bounds = np.array([env_params.codesign.low])
upper_bounds = np.array([env_params.codesign.high])
dim = 6

# Generate grid of models
n_pts = 256
designs = model_lib.sample_designs(
        rng         = np.random.default_rng(0),
        num_envs    = n_pts,
        low         = lower_bounds,
        high        = upper_bounds,
        dim         = dim,
    )

models = model_lib.build_batched_model(env, designs)
_, reset = mm.get_step_reset(env)

# Load value inference
value_inference_fn, val_params = load_design_value_hypernetwork(config)
state = jax.vmap(reset, in_axes=(None, 0))(0, models)
# Load the trained, design-conditioned value network
value_fn = value_inference_fn(val_params, designs)
obs = jax.vmap(env._get_obs)(state.data, state.info)
values_from_val_fn = value_fn(obs)


inference_fn, policy_params = load_design_hypernetwork(config)

base_policy = inference_fn(policy_params, model_lib.normalize_design(designs, lower_bounds, upper_bounds), deterministic=True)
policy = policy_lib.from_inference_fn(base_policy)
rewards_rollout = rollout_so_parallel(
    env,
    models,
    policy,
    n_pts,
    config.learning_params.ppo_params.episode_length,
    mask_after_done=True,
    seed=0,
)

discount = config.learning_params.ppo_params.discounting
total_rewards_rollout = np.sum(rewards_rollout*np.pow(discount, np.arange(config.learning_params.ppo_params.episode_length))[:, np.newaxis], axis=0)

plt.title("Predicted vs Actual total rewards")
plt.scatter(values_from_val_fn, total_rewards_rollout)
plt.xlabel("Predicted reward from value function")
plt.ylabel("Measured reward from rollout")
plt.show()