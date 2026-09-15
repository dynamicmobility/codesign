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
from scripts import icra
from dynamo_figures import CompositeMode, CompositeImage

RUN_ID = icra.FINAL_CONFIGS['walker']['MDH'].run_id
CONFIG_PATH = f"results/wandb-downloads/{RUN_ID}/config.yaml"

# Load in the dataset into a grid
grid = codesign.Grid.load(f"scripts/icra/outputs/{RUN_ID}/nsga3_front.npz")
TRADEOFF_LABELS = ['Run Reward', 'Energy Reward', 'Height Reward']
CMAP = 'plasma'
STAMP_CMAP = 'peak'
STAMP_X_LABEL = 'Left Length'
STAMP_Y_LABEL = 'Right Length'

GEN_VIDEO = False


ax = plt.axes()
rewards = grid.rewards.reshape((-1,) + grid.rewards.shape[3:])
# TODO: Project this down into 3 dimensions or less
designs = grid.designs.reshape((-1, grid.designs.shape[2]))
tradeoffs = grid.tradeoffs.reshape((-1, grid.tradeoffs.shape[2]))
designs_color = np.hstack((np.sum(designs[:, 0:3], axis=1)[:, None], np.sum(designs[:, 3:], axis=1)[:, None]))
colors, stamp = colorstamps.apply_stamp(designs_color[:, 0], designs_color[:, 1], STAMP_CMAP)

nd_idx = get_nondominated(rewards)
designs_nd = designs[nd_idx, :]
tradeoffs_nd = tradeoffs[nd_idx, :]

tradeoffs_d = np.array([[1, 0], [0.0, 1], [.1, 1]])

idxs = (tradeoffs_d @ rewards[nd_idx, :].T).argmax(axis=1)

full_idxs = np.arange(rewards.shape[0])[nd_idx][idxs]

def rgba_to_hex(rgba):
    """Convert four RGBA values in [0, 1] to a #RRGGBBAA hex color."""
    return matplotlib.colors.to_hex(rgba, keep_alpha=True)


print(f"Design 1: {designs_nd[idxs[0], :]}, Tradeoff 1: {tradeoffs_nd[idxs[0], :]}, Color 1: {rgba_to_hex(np.concat((colors[full_idxs[0], :], np.array([1.0]))))}")
print(f"Design 2: {designs_nd[idxs[1], :]}, Tradeoff 2: {tradeoffs_nd[idxs[1], :]}, Color 2: {rgba_to_hex(np.concat((colors[full_idxs[1], :], np.array([1.0]))))}")
print(f"Design 3: {designs_nd[idxs[2], :]}, Tradeoff 3: {tradeoffs_nd[idxs[2], :]}, Color 3: {rgba_to_hex(np.concat((colors[full_idxs[2], :], np.array([1.0]))))}")

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

settings = [
    {"mode": CompositeMode.MIN_VALUE, "right_crop": [0, 640], "t_end": 1.2, "camera": "side_fixed", "top_crop": [50, 370]},
    {"mode": CompositeMode.MIN_VALUE, "right_crop": [0, 640], "t_end": 1.2, "camera": "side_fixed", "top_crop": [50, 370]},
    {"mode": CompositeMode.MIN_VALUE, "right_crop": [0, 640], "t_end": 1.2, "camera": "side_fixed", "top_crop": [50, 370]},
]

for i in range(len(idxs)):
    filename = f'scripts/icra/outputs/{RUN_ID}/design{i}.mp4'
    setting = settings[i]
    merger = CompositeImage(
        mode=setting["mode"],
        video_path=filename,
        start_t=0.0,
        end_t=setting["t_end"],
        skip_frame=10,
        alpha=0.15
    )
    # Generate the composite
    top = setting["top_crop"][0]
    bottom = setting["top_crop"][1]
    left = setting["right_crop"][0]
    right = setting["right_crop"][1]
    result = merger.merge_images()
    result = result[top:bottom, left:right][:, :, [2, 1, 0]]
    plt.imsave(f'scripts/icra/outputs/{RUN_ID}/design{i}.png', result)


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

stamp_ax = stamp.overlay_ax(ax, lower_left_corner=[0.35, 0.35], width=0.2)
stamp_ax.set_xlabel(STAMP_X_LABEL)
stamp_ax.set_ylabel(STAMP_Y_LABEL)
codesign.dress_axis(ax)
codesign.dress_axis(stamp_ax)
plt.tight_layout()

plt.savefig(f"scripts/icra/outputs/{RUN_ID}/pareto.pdf")
plt.savefig(f"scripts/icra/outputs/{RUN_ID}/pareto.png", dpi=600)
plt.show()
