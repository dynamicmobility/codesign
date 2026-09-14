import codesign
import colorstamps
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from itertools import combinations
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated
import minimal_mjx as mm
matplotlib.use("TKAgg")

RUN_ID = "zpzu9ms5"
CONFIG_PATH = "results/wandb-downloads/zpzu9ms5/config.yaml"

# TODO: Check if a dataset with the name {run_id}/sweep.npz exists in scripts/icra/data
# If it does not, use generate_data to generate it.

# Load in the dataset into a grid
grid = codesign.Grid.load(f"scripts/icra/outputs/{RUN_ID}/nsga3_front.npz")
TRADEOFF_LABELS = ['Run Reward', 'Energy Reward', 'Height Reward']
CMAP = 'plasma'
STAMP_CMAP = 'peak'
STAMP_X_LABEL = 'Front Length'
STAMP_Y_LABEL = 'Back Length'

GEN_VIDEO = True


ax = plt.axes(projection="3d")
rewards = grid.rewards.reshape((-1,) + grid.rewards.shape[3:])
# TODO: Project this down into 3 dimensions or less
designs = grid.designs.reshape((-1, grid.designs.shape[2]))
tradeoffs = grid.tradeoffs.reshape((-1, grid.tradeoffs.shape[2]))
designs_color = np.hstack((np.sum(designs[:, 0:3], axis=1)[:, None], np.sum(designs[:, 3:], axis=1)[:, None]))
colors, stamp = colorstamps.apply_stamp(designs_color[:, 0], designs_color[:, 1], STAMP_CMAP)

nd_idx = get_nondominated(rewards)
designs_nd = designs[nd_idx, :]
tradeoffs_nd = tradeoffs[nd_idx, :]

tradeoffs_d = np.array([[1, 0, 0], [0.01, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1]])

idxs = (tradeoffs_d @ rewards[nd_idx, :].T).argmax(axis=1)

full_idxs = np.arange(rewards.shape[0])[nd_idx][idxs]

print(f"Design 1: {designs_nd[idxs[0], :]}, Tradeoff 1: {tradeoffs_nd[idxs[0], :]}")
print(f"Design 2: {designs_nd[idxs[1], :]}, Tradeoff 2: {tradeoffs_nd[idxs[1], :]}")
print(f"Design 3: {designs_nd[idxs[2], :]}, Tradeoff 3: {tradeoffs_nd[idxs[2], :]}")

if(GEN_VIDEO):
    for i in range(len(idxs)):
        config = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
        codesign.save_policy_rollout_video(
            config,
            out_path = f'scripts/icra/outputs/{RUN_ID}/design{i}.mp4',
            design=designs_nd[idxs[i]],
            tradeoff=tradeoffs_nd[idxs[i]],
            n_steps=200,
            camera='side_fixed',
            use_caption=False,
            robot_color=np.concat((colors[full_idxs[i], :], np.array([1.0])))
        )

mop.plot_pareto(
    ax=ax,
    pareto=rewards,
    objective = TRADEOFF_LABELS,
    colors=colors,
    dominated_alpha = 0.2,
    dominated_s = 8,
    nondominated_s  = 30,
    outline_nondominated=1.0,
    show_dominated=True,
    connect=False,
    marker='o',
    special_idxs = full_idxs,
    special_marker = "*",
)

stamp_ax = stamp.overlay_ax(ax, lower_left_corner=[0.85, 0.0], width=0.2)
stamp_ax.set_xlabel(STAMP_X_LABEL)
stamp_ax.set_ylabel(STAMP_Y_LABEL)
ax.view_init(elev=30, azim=45)
ax.xaxis.set_rotate_label(True)
codesign.dress_axis(ax)
codesign.dress_axis(stamp_ax)

plt.savefig(f"scripts/icra/outputs/{RUN_ID}/pareto.pdf")
plt.savefig(f"scripts/icra/outputs/{RUN_ID}/pareto.png", dpi=600)
plt.show()
