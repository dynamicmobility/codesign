from dataclasses import dataclass
from typing import Optional

from codesign.utils import model as model_lib
from codesign.eval.parallel_eval import rollout_so_parallel
from minimal_mjx.eval import policy as policy_lib
from codesign.learning.inference import load_design_hypernetwork, load_design_value_hypernetwork, load_mo_design_value_hypernetwork
from codesign.learning.inference import load_mo_design_hypernetwork, load_mo_design_predictor_hypernetwork
from functools import partial

import minimal_mjx as mm
import moplayground as mop
import codesign
import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TKAgg")


def load_value_network(config):
    algorithm = config["algorithm"]
    if algorithm == "design_hypernetwork":
        return load_design_value_hypernetwork(config)
    elif algorithm in ("mo_design_hypernetwork", "mo_design_predictor_hypernetwork"):
        return load_mo_design_value_hypernetwork(config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}")


def load_policy_network(config):
    algorithm = config["algorithm"]
    if algorithm == "design_hypernetwork":
        return load_design_hypernetwork(config)
    elif algorithm == "mo_design_hypernetwork":
        return load_mo_design_hypernetwork(config)
    elif algorithm == "mo_design_predictor_hypernetwork":
        inference_fn, _, params = load_mo_design_predictor_hypernetwork(config)
        return inference_fn, params
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}")


# CONFIG_PATH = "config/design_hypernetwork/cheetah6D.yaml"
CONFIG_PATH = "results/wandb-downloads/bkgp6dfy/config.yaml"
DEFAULT_TRADEOFF = jnp.array([1.0, 1.0, 1.0])
config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
env, env_params = codesign.load_env(config=config, backend="jnp")
lower_bounds = np.array([env_params.codesign.low])
upper_bounds = np.array([env_params.codesign.high])
dim = len(upper_bounds)

# Generate grid of models
n_pts = 64
designs = model_lib.sample_designs(
        rng         = np.random.default_rng(0),
        num_envs    = n_pts,
        low         = lower_bounds,
        high        = upper_bounds,
        dim         = dim,
    )

models = model_lib.build_batched_model(env, designs)
_, reset = mm.get_step_reset(env)
tradeoff_args = (
    (jnp.broadcast_to(DEFAULT_TRADEOFF, (n_pts, DEFAULT_TRADEOFF.size)),)
    if config["algorithm"] in ("mo_design_hypernetwork", "mo_design_predictor_hypernetwork")
    else ()
)

# Load value inference
value_inference_fn, val_params = load_value_network(config)
state = jax.vmap(reset, in_axes=(None, 0))(0, models)
# Load the trained, design-conditioned value network
value_fn = value_inference_fn(val_params, designs, *tradeoff_args)
obs = jax.vmap(env._get_obs)(state.data, state.info)
values_from_val_fn = value_fn(obs)
if tradeoff_args:
    values_from_val_fn = np.sum(values_from_val_fn * DEFAULT_TRADEOFF, axis=-1)


inference_fn, policy_params = load_policy_network(config)

base_policy = inference_fn(policy_params, model_lib.normalize_design(designs, lower_bounds, upper_bounds), *tradeoff_args, deterministic=True)
policy = policy_lib.from_inference_fn(base_policy)
rewards_rollout = rollout_so_parallel(
    codesign.CodesignMO2SO(env, DEFAULT_TRADEOFF) if tradeoff_args else env,
    models,
    policy,
    n_pts,
    config.learning_params.ppo_params.episode_length,
    mask_after_done=True,
    seed=0,
)

discount = config.learning_params.ppo_params.discounting
discounts = discount ** np.arange(rewards_rollout.shape[0])
total_rewards_rollout = np.sum(rewards_rollout * discounts[:, None], axis=0)
plt.title("Predicted vs Actual total rewards")
plt.scatter(values_from_val_fn, total_rewards_rollout)
plt.xlabel("Predicted reward from value function")
plt.ylabel("Measured reward from rollout")
plt.show()
