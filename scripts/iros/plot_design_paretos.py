import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from itertools import combinations
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

TRADEOFF_LABELS = ['Run', 'Energy', 'Height']
DESIGN_LABEL = 'Back leg scale'
CMAP = 'viridis'

def main():
    fig, axs = plt.subplots(nrows=3, figsize=(9, 15), layout='constrained')
    axs = axs.flatten()
    usup = codesign.DesignTradeoffDataset.load('scripts/outputs/datasets/ls8xn17y/sweep.npz')
    # usup = codesign.DesignTradeoffDataset.load('scripts/outputs/datasets/sweep-j24wm6rn-best-again/sweep.npz')

    for i, (ax, objs) in enumerate(zip(axs, combinations(range(3), 2))):
        # rewards, designs, tradeoffs = usup.flatten(tradeoff_slice=objs)
        rewards, designs, tradeoffs = usup.flatten()
        norm = Normalize(vmin=designs.min(), vmax=designs.max())
        colors = codesign.get_colors(norm(designs), cmap=CMAP)
        mop.plot_pareto(
            ax        = ax,
            pareto    = rewards[:, objs],
            colors    = colors[:, 0, :],
            objective = [TRADEOFF_LABELS[i] for i in objs],
        )
        ax.set_title(f'Tradeoffs = {', '.join([TRADEOFF_LABELS[i] for i in objs])}')

    fig.colorbar(
        ScalarMappable(norm=norm, cmap=CMAP), ax=axs, label=DESIGN_LABEL, aspect=50
    )
    fig.savefig('scripts/iros/outputs/dsup_vs_usup.svg')

if __name__ == '__main__':
    main()
