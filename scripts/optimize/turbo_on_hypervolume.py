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
from codesign.utils.grid import Grid, sample_tradeoffs_cpu
from codesign.eval.parallel_eval import rollout_so_parallel, build_grid_rollout_fn, grid_keys
from minimal_mjx.eval import policy as policy_lib
from codesign.learning.inference import load_design_hypernetwork, load_mo_design_hypernetwork
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import minimal_mjx as mm
import moplayground as mop
import codesign
import numpy as np
import jax.numpy as jnp
import jax
import time
matplotlib.use('tkagg')


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# This is the best design: 1.02006672 1.71947823 1.94088529 1.47456715 0.90478917 1.47590946
CONFIG_PATH = "results/wandb-downloads/zpzu9ms5/config.yaml"
ROLLOUT_STEPS = 200
NUM_TRADEOFFS = 32

config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
# env        = codesign.cheetah(env_params=env_params, backend="jnp")
env, env_params = codesign.load_env(config=config, backend="jnp")

lower_bounds = jnp.array([env_params.codesign.low])
upper_bounds = jnp.array([env_params.codesign.high])

batch_size=16
dim = lower_bounds.size
n_init = 16
max_cholesky_size = float("inf")  # Always use Cholesky


inference_fn, params = load_mo_design_hypernetwork(config)

rollout = build_grid_rollout_fn(env, ROLLOUT_STEPS, inference_fn, deterministic=True)
tradeoffs = sample_tradeoffs_cpu(
    np.random.default_rng(0),
    32,
    len(env.objectives),
    sampling="sparse-heavytail"
)

# In maximization/reward space; choose a fixed reference below relevant returns.
ref_point = np.zeros(tradeoffs.shape[-1])


def rollout_fun(designs_normalized: torch.Tensor):
    designs = model_lib.unnormalize_design(jnp.array(designs_normalized.numpy()), low=lower_bounds, high=upper_bounds)

    grid = Grid.crossed(designs, tradeoffs)
    # Models carry (num_designs, num_tradeoffs) axes.
    # Only unique physical designs are compiled.
    models = grid.build_models(env, tiled=False)

    policy_designs = model_lib.normalize_design(
        jnp.asarray(grid.designs),
        config=config,
    )

    (_, _, returns), _ = rollout(
        grid_keys(grid, seed=0),
        policy_designs,
        jnp.asarray(grid.tradeoffs),
        models,
        params,
    )
    # returns: (num_designs, num_tradeoffs, per_cell, num_objectives)
    mean_returns = np.asarray(returns).mean(axis=2)

    # Each design's tradeoffs produce a separate set of objective vectors.
    hypervolumes = [
        mop.get_pareto_statistics(front, ref_point=ref_point)[0]
        for front in mean_returns
    ]

    # TuRBO / SingleTaskGP expects one output column.
    ret =  torch.tensor(
        hypervolumes,
        dtype=designs_normalized.dtype,
        device=designs_normalized.device,
    ).reshape(-1, 1)
    return ret




def get_initial_points(dim: int, n_pts: int, seed: int = 0) -> torch.Tensor:
    """Generate initial points in normalized design space using Sobol sequence."""
    return torch.tensor(np.atleast_2d(model_lib.sample_designs(
        rng         = np.random.default_rng(seed),
        num_envs    = n_pts,
        low         = jnp.zeros((dim,)),
        high        = jnp.ones((dim,)),
        dim         = dim,
    )).astype(np.double))


# X_turbo = get_initial_points(dim, n_init)
optim = TurboOptimizer(
    dim=dim,
    fun=rollout_fun,
    max_cholesky_size=max_cholesky_size,
    batch_size=batch_size,
)

designs = get_initial_points(dim, n_init)
assert torch.any((designs > 1) | (designs < 0))==False


best_design, best_value, X_turbo, Y_turbo = optim.optimize(initial_guess=designs)

print("Best Design: ", model_lib.unnormalize_design(best_design.numpy(), low=lower_bounds, high=upper_bounds))
print("Best Value: ", best_value)
print("Re-evaluated value:", rollout_fun(best_design.unsqueeze(0)))
print("Re-evaluated again:", rollout_fun(best_design.unsqueeze(0)))

eval_designs = model_lib.sample_designs(np.random.default_rng(0), 32, low=lower_bounds, high=upper_bounds, dim=6)
eval_grid = model_lib.normalize_design(eval_designs, low=lower_bounds, high=upper_bounds)
print("Other Rollouts: ", rollout_fun(torch.tensor(eval_grid)))


names = ["TuRBO-1"]
runs = [Y_turbo]
fig, ax = plt.subplots(figsize=(8, 6))

for _name, run in zip(names, runs, strict=True):
    fx = np.maximum.accumulate(run.cpu())
    plt.plot(fx, marker="", lw=3)

plt.xlabel("Function value", fontsize=18)
plt.xlabel("Number of evaluations", fontsize=18)
plt.title("Best Hypervolume", fontsize=24)
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