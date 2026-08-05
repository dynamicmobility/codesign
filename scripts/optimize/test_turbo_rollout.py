import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch.quasirandom import SobolEngine

from botorch.exceptions import BadInitialCandidatesWarning
from botorch.test_functions import Ackley

from codesign.optimizers import TurboState, TurboOptimizer
from codesign.utils import model as model_lib
from codesign.eval.parallel_eval import rollout_so_parallel
from minimal_mjx.eval import policy as policy_lib
from codesign.learning.inference import load_design_hypernetwork, load_mo_design_hypernetwork
from functools import partial

import minimal_mjx as mm
import moplayground as mop
import codesign
import numpy as np
import jax.numpy as jnp
import jax
import time


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONFIG_PATH = "config/design_hypernetwork/cheetah6D.yaml"
config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
# env        = codesign.cheetah(env_params=env_params, backend="jnp")
env, env_params = codesign.load_env(config=config, backend="jnp")

lower_bounds = jnp.array([env_params.codesign.low])
upper_bounds = jnp.array([env_params.codesign.high])

batch_size=4
dim = 6
n_init = 2 * dim
max_cholesky_size = float("inf")  # Always use Cholesky


inference_fn, params = load_design_hypernetwork(config)

def rollout_fun(designs_normalized: torch.Tensor):
    designs = model_lib.unnormalize_design(jnp.array(designs_normalized.numpy()), low=lower_bounds, high=upper_bounds)
    num_envs = designs_normalized.shape[0]

    # One stacked, batched mjx.Model per design (host-side, via the env's generator).
    batched_model = model_lib.build_batched_model(env, designs)

    # Load the trained, design-conditioned policy (obs/action sizes come from the
    # checkpoint's saved config) and adapt it to the (obs, key, t) rollout protocol.
    base_policy = inference_fn(params, designs_normalized, deterministic=True)
    policy = policy_lib.from_inference_fn(base_policy)

    rewards = rollout_so_parallel(
        env,
        batched_model,
        policy,
        num_envs,
        config.learning_params.ppo_params.episode_length,
        mask_after_done=True,
        seed=0,
    )
    return torch.from_numpy(np.atleast_2d(rewards.sum(axis=0)).T.astype(np.double).copy())

def get_initial_points(dim: int, n_pts: int, seed: int = 0) -> jnp.array:
    """Generate initial points in normalized design space using Sobol sequence."""
    return np.atleast_2d(model_lib.sample_designs(
        rng         = np.random.default_rng(seed),
        num_envs    = n_pts,
        low         = jnp.zeros((dim,)),
        high        = jnp.ones((dim,)),
        dim         = dim,
    )).astype(np.double)


# X_turbo = get_initial_points(dim, n_init)
optim = TurboOptimizer(
    dim=dim,
    fun=rollout_fun,
    max_cholesky_size=max_cholesky_size,
    batch_size=batch_size,
)

designs = get_initial_points(dim, n_init)
X_next, X_turbo, Y_next, Y_turbo = optim.optimize(initial_guess=designs)

print(X_next)
print(Y_next)

import matplotlib.pyplot as plt
# import numpy as np


names = ["TuRBO-1"]
runs = [Y_turbo]
fig, ax = plt.subplots(figsize=(8, 6))

for _name, run in zip(names, runs, strict=True):
    fx = np.maximum.accumulate(run.cpu())
    plt.plot(fx, marker="", lw=3)

plt.xlabel("Function value", fontsize=18)
plt.xlabel("Number of evaluations", fontsize=18)
plt.title("20D Ackley", fontsize=24)
plt.xlim([0, len(Y_turbo)])

plt.grid(visible=True)
plt.tight_layout()
plt.legend(
    [*names],
    loc="lower center",
    bbox_to_anchor=(0, -0.08, 1, 1),
    bbox_transform=plt.gcf().transFigure,
    ncol=5,
    fontsize=16,
)
plt.show()