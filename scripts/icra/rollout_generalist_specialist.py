import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from itertools import combinations
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated
from codesign.eval.parallel_eval import rollout_so_parallel, build_grid_rollout_fn, grid_keys
from codesign.learning.inference import load_mo_design_hypernetwork
import jax.numpy as jnp
from codesign.utils.grid import Grid, sample_tradeoffs_cpu
import codesign.utils.model as model_lib


CONFIG_PATH = "results/wandb-downloads/zpzu9ms5/config.yaml"
config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
# env        = codesign.cheetah(env_params=env_params, backend="jnp")
env, env_params = codesign.load_env(config=config, backend="jnp")
lower_bounds = np.array([env_params.codesign.low])
upper_bounds = np.array([env_params.codesign.high])
ROLLOUT_STEPS = 500

# This is obtained by running scripts/optimize/turbo_on_hypervolume.py
GENERALIST_DESIGN = [1.02006672, 1.71947823, 1.94088529, 1.47456715, 0.90478917, 1.47590946]
# These are obtained from plot_pareto_3d
RUN_DESIGN = [0.6152434, 0.8852895, 1.4163877, 1.1994246, 1.0384812, 0.52137005]
ENERGY_DESIGN = [1.293534, 0.66744065, 0.66819084, 0.55856186, 1.9247661, 1.0071616 ]
HEIGHT_DESIGN = [1.3124598, 1.9924996, 1.8101603, 1.8071598, 1.522273, 0.7453903]


inference_fn, params = load_mo_design_hypernetwork(config)
rollout = build_grid_rollout_fn(env, ROLLOUT_STEPS, inference_fn, deterministic=True)
tradeoffs = sample_tradeoffs_cpu(
    np.random.default_rng(0),
    128,
    len(env.objectives),
    sampling="sparse-heavytail"
)

output_dir = Path("scripts/icra/outputs/zpzu9ms5")
output_dir.mkdir(parents=True, exist_ok=True)

for name, design in (
    ("generalist", GENERALIST_DESIGN),
    ("run", RUN_DESIGN),
    ("energy", ENERGY_DESIGN),
    ("height", HEIGHT_DESIGN),
):
    grid = Grid.crossed(design, tradeoffs)
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

    grid.rewards = np.asarray(returns)
    grid.objectives = list(env.objectives)
    grid.save(output_dir / f"{name}.npz")
