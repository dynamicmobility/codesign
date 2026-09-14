import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from itertools import combinations
from pathlib import Path
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated
from codesign.eval.parallel_eval import rollout_so_parallel, build_grid_rollout_fn, grid_keys
from codesign.learning.inference import load_mo_design_hypernetwork
import jax.numpy as jnp
from codesign.utils.grid import Grid, sample_tradeoffs_cpu
import codesign.utils.model as model_lib
matplotlib.use('tkagg')


CONFIG_PATH = "results/wandb-downloads/zpzu9ms5/config.yaml"
config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
# env        = codesign.cheetah(env_params=env_params, backend="jnp")
env, env_params = codesign.load_env(config=config, backend="jnp")
lower_bounds = np.array([env_params.codesign.low])
upper_bounds = np.array([env_params.codesign.high])
ROLLOUT_STEPS = 200

# This is obtained by running scripts/optimize/turbo_on_hypervolume.py
GENERALIST_DESIGN = [1.02006672, 1.71947823, 1.94088529, 1.47456715, 0.90478917, 1.47590946]
# These are obtained from plot_pareto_3d
RUN_DESIGN = [0.6152434, 0.8852895, 1.4163877, 1.1994246, 1.0384812, 0.52137005]
ENERGY_DESIGN = [1.293534, 0.66744065, 0.66819084, 0.55856186, 1.9247661, 1.0071616 ]
HEIGHT_DESIGN = [1.3124598, 1.9924996, 1.8101603, 1.8071598, 1.522273, 0.7453903]

OBJECTIVE_LABELS = ["Run", "Energy", "Height"]

output_dir = Path("scripts/icra/outputs/zpzu9ms5")
output_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure()
ax_3d = fig.add_subplot(2,2,1, projection='3d')
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

axs = [ax2, ax3, ax4]

# TODO: Get colors of corner designs from plot_pareto_3d

for name, design, color in (
    ("generalist", GENERALIST_DESIGN, "#000000"),
    ("run", RUN_DESIGN, "#00FFFF"),
    ("energy", ENERGY_DESIGN, "#008080"),
    ("height", HEIGHT_DESIGN, "#C6AFE9"),
):
    grid = codesign.Grid.load(output_dir / f"{name}.npz")

    rewards = grid.rewards.reshape((-1,) + grid.rewards.shape[3:])
    # TODO: Plot 2D frontiers too using the procedure from plot_nsga2_front
    for i, objs in enumerate(combinations(range(grid.n_r), 2)):
        mop.plot_pareto(
            pareto = rewards[:, objs],
            ax=axs[i],
            colors=color,
            objective=[OBJECTIVE_LABELS[o] for o in objs],
            connect = True,
            show_dominated=True,
            dominated_alpha=0.2,
            nondominated_alpha=1,
            label=name,
            set_lims=False
        )

    mop.plot_pareto(
        ax_3d, 
        rewards,
        colors=color,
        objective=OBJECTIVE_LABELS,
        connect = False,
        show_dominated=True,
        dominated_alpha=0.2,
        nondominated_alpha=1,
        label=name,
        set_lims=False
    )

ax_3d.legend()
plt.show()