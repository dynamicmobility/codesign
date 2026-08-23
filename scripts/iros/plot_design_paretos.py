import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from itertools import combinations
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

DATASET = 'scripts/outputs/datasets/ls8xn17y/sweep_2d.npz'
# DATASET = 'scripts/outputs/datasets/sweep-j24wm6rn-best-again/sweep.npz'
OUT_DIR = 'scripts/iros/outputs'
TRADEOFF_LABELS = ['Run', 'Energy', 'Height']
DESIGN_LABEL = 'Back leg scale'
CMAP = 'plasma'

DESIGN_A = 0.5

DESIGN_B = 2.0

def main():
    usup = codesign.DesignTradeoffDataset.load(DATASET)
    num_obj = usup.rewards.shape[-1]
    norm = Normalize(vmin=usup.designs.min(), vmax=usup.designs.max())

    for objs in combinations(range(num_obj), 2):
        # the pair's 2D face: tradeoffs putting no weight on the objective left out
        left_out = (set(range(num_obj)) - set(objs)).pop()
        on_face  = np.isclose(usup.tradeoffs[:, left_out], 0, atol=1e-3)
        rewards  = usup.mean_rewards[:, on_face].reshape(-1, num_obj)
        designs  = np.repeat(usup.designs, on_face.sum(), axis=0)

        fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
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
            connect=True
        )

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
                set_lims=False
            )
        
        ax = codesign.dress_axis(ax)
        # ax.set_title(f"Desi{' vs. '.join(TRADEOFF_LABELS[o] for o in objs)}")
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=CMAP), ax=ax, label=DESIGN_LABEL, aspect=50
        )
        name = '_'.join(TRADEOFF_LABELS[o].lower() for o in objs)
        fig.savefig(f'{OUT_DIR}/dsup_vs_usup_{name}.svg')
        plt.close(fig)

if __name__ == '__main__':
    main()
