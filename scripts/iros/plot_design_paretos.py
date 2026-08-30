import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from itertools import combinations
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated

DATASET = 'scripts/outputs/datasets/95mhoslv/sweep_2d.npz'
# DATASET = 'scripts/outputs/datasets/sweep-j24wm6rn-best-again/sweep.npz'
OUT_DIR = 'scripts/iros/outputs'
TRADEOFF_LABELS = ['Run', 'Energy', 'Height']
DESIGN_LABEL = 'Back leg scale'
CMAP = 'plasma'

DESIGN_A = 0.5

DESIGN_B = 2.0

def main():
    usup = codesign.Grid.load(DATASET)
    num_obj = usup.rewards.shape[-1]
    norm = Normalize(vmin=usup.designs.min(), vmax=usup.designs.max())

    for objs in combinations(range(num_obj), 2):
        # the pair's 2D face: tradeoffs putting no weight on the objective left out
        left_out = (set(range(num_obj)) - set(objs)).pop()
        on_face  = np.isclose(usup.unique_tradeoffs[:, left_out], 0, atol=1e-3)
        rewards  = usup.mean_rewards[:, on_face].reshape(-1, num_obj)
        designs  = usup.designs[:, on_face].reshape(-1, usup.design_dim)

        fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
        # one frontier per called-out design, coloured on the same scale as the colorbar
        for d in (DESIGN_A, DESIGN_B):
            on_design = np.isclose(d, designs, atol=1e-1).ravel()
            mop.plot_pareto(
                ax        = ax,
                pareto    = rewards[on_design][:, objs],
                colors    = codesign.get_colors(norm(designs[on_design]), cmap=CMAP)[:, 0, :],
                objective = [TRADEOFF_LABELS[o] for o in objs],
                show_dominated=False,
                connect=True,
                nondominated_s  = 30,
                outline_nondominated=1.0,
                set_lims=False,
                marker='s',
            )
        
        mop.plot_pareto(
            ax        = ax,
            pareto    = rewards[:, objs],
            colors    = codesign.get_colors(norm(designs), cmap=CMAP)[:, 0, :],
            objective = [TRADEOFF_LABELS[o] for o in objs],
            dominated_alpha = 0.2,
            dominated_s = 8,
            nondominated_s  = 30,
            outline_nondominated=1.0,
            show_dominated=False,
            connect=True,
            marker='o'
        )

        nd_idx = get_nondominated(rewards[:, objs])
        nd_rewards = rewards[:, objs][nd_idx]
        rewards_sorted = np.sort(np.sum(rewards[:, objs][nd_idx], axis=1))
        # print(rewards_sorted[int(rewards_sorted.size/2)])
        nd_designs = designs[nd_idx, :]
        print(f"In {TRADEOFF_LABELS[objs[0]]} vs {TRADEOFF_LABELS[objs[1]]}, the " \
              f"{TRADEOFF_LABELS[objs[0]]} maximizing design is {nd_designs[np.argmax(nd_rewards[:, 0])]} " \
              f"the {TRADEOFF_LABELS[objs[1]]} maximizing design is {nd_designs[np.argmax(nd_rewards[:, 1])]} " \
              f"and the mixed maximizing design is {nd_designs[np.argmax(nd_rewards[:, 0] + nd_rewards[:, 1])]}")

        # print(f"Color for {nd_designs[np.argmax(nd_rewards[:, 0])]} is {codesign.get_colors(norm(designs), cmap=CMAP)[:, nd_designs[np.argmax(nd_rewards[:, 0])], :]}")
        # print(f"Color for {nd_designs[np.argmax(nd_rewards[:, 1])]} is {codesign.get_colors(norm(designs), cmap=CMAP)[:, nd_designs[np.argmax(nd_rewards[:, 1])], :]}")
        # print(f"Color for {nd_designs[np.argmax(nd_rewards[:, 0] + nd_rewards[:, 1])]} is {codesign.get_colors(norm(designs), cmap=CMAP)[:, nd_designs[np.argmax(nd_rewards[:, 0] + nd_rewards[:, 1])], :]}")

        ax = codesign.dress_axis(ax)
        # ax.set_title(f"Desi{' vs. '.join(TRADEOFF_LABELS[o] for o in objs)}")
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=CMAP), ax=ax, label=DESIGN_LABEL, aspect=50
        )
        name = '_'.join(TRADEOFF_LABELS[o].lower() for o in objs)
        fig.savefig(f'{OUT_DIR}/design_pareto_{name}.svg')
        plt.close(fig)

if __name__ == '__main__':
    main()
