import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from itertools import combinations
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated

MDH = 'results/Sep08/mo-up-cheetah6d-trial/sweep_2d.npz'
MORLAX = 'results/Sep10/test/sweep_2d.npz'
# DATASET = 'scripts/outputs/datasets/sweep-j24wm6rn-best-again/sweep.npz'
OUT_DIR = 'scripts/icra/outputs'
TRADEOFF_LABELS = ['Run', 'Energy', 'Height']
DESIGN_LABEL = 'Back leg scale'
CMAP = 'plasma'

DESIGN_A = np.ones(6)

def get_2d_fronts(grid: codesign.Grid):
    num_obj = grid.rewards.shape[-1]
    rewards, designs = {}, {}
    for objs in combinations(range(num_obj), 2):
        # the pair's 2D face: tradeoffs putting no weight on the objective left out
        left_out = (set(range(num_obj)) - set(objs)).pop()
        on_face  = np.isclose(grid.unique_tradeoffs[:, left_out], 0, atol=1e-3)
        rewards[objs] = grid.mean_rewards[:, on_face].reshape(-1, num_obj)
        designs[objs] = grid.designs[:, on_face].reshape(-1, grid.design_dim)
        
    return rewards, designs

def main():
    mdh = codesign.Grid.load(MDH)
    morlax = codesign.Grid.load(MORLAX)
    # quit()
    num_obj = mdh.rewards.shape[-1]
    # norm = Normalize(vmin=mdh.designs.min(), vmax=mdh.designs.max())
    
    mdh_rewards, mdh_designs = get_2d_fronts(mdh)
    morlax_rewards, morlax_designs = get_2d_fronts(morlax)

    for objs in combinations(range(num_obj), 2):
        # the pair's 2D face: tradeoffs putting no weight on the objective left out

        fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
        # one frontier per called-out design, coloured on the same scale as the colorbar
        # for d in (DESIGN_A,):
        #     on_design = np.isclose(d, designs, atol=1e-1).ravel()
        #     mop.plot_pareto(
        #         ax        = ax,
        #         pareto    = rewards[on_design][:, objs],
        #         colors    = codesign.get_colors(norm(designs[on_design]), cmap=CMAP)[:, 0, :],
        #         objective = [TRADEOFF_LABELS[o] for o in objs],
        #         show_dominated=False,
        #         connect=True,
        #         nondominated_s  = 30,
        #         outline_nondominated=1.0,
        #         set_lims=False,
        #         marker='s',
        #     )
        back_leg_lengths = np.sum(mdh_designs[objs][:, 3:], axis=1).reshape(-1, 1)
        norm = Normalize(vmin=back_leg_lengths.min(), vmax=back_leg_lengths.max())
        print(back_leg_lengths.min(), back_leg_lengths.max())
        mop.plot_pareto(
            ax        = ax,
            pareto    = mdh_rewards[objs][:, objs],
            colors    = codesign.get_colors(norm(back_leg_lengths), cmap=CMAP)[:, 0, :],
            objective = [TRADEOFF_LABELS[o] for o in objs],
            dominated_alpha = 0.2,
            dominated_s = 8,
            nondominated_s  = 30,
            outline_nondominated=1.0,
            show_dominated=True,
            connect=True,
            marker='o',
            set_lims=False
        )
        
        back_leg_lengths = np.sum(morlax_designs[objs][:, 3:], axis=1).reshape(-1, 1)
        mop.plot_pareto(
            ax        = ax,
            pareto    = morlax_rewards[objs][:, objs],
            colors    = codesign.get_colors(norm(back_leg_lengths), cmap=CMAP)[:, 0, :],
            objective = [TRADEOFF_LABELS[o] for o in objs],
            dominated_alpha = 0.2,
            dominated_s = 8,
            nondominated_s  = 30,
            outline_nondominated=1.0,
            show_dominated=False,
            connect=True,
            marker='o',
            set_lims=False
        )

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
