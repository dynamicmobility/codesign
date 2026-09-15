"""Overlay the design Pareto fronts each algorithm searched out for one robot.

Every algorithm's front is the ``nsga3_front.npz`` that ``nsga2_design_front.py`` wrote
under its :class:`icra.Run`'s ``dataset_path/<run id>/`` -- the non-dominated
``(design, tradeoff)`` pairs of that run's trained network. This draws one figure per
objective pair, with every algorithm's front on the same axes in its own colour, so the
fronts can be read against each other. An algorithm with no run configured, or whose
dataset has not been generated yet, is warned about and skipped.

Run from the repository root, where the configured dataset paths resolve:
``python -m scripts.icra.subopt --robot cheetah``.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from moplayground.utils.pareto import get_nondominated

import codesign

from scripts import icra

OUT_DIR    = "scripts/icra/outputs"
FRONT_NAME = "nsga3_front.npz"

# The algorithms overlaid, each with its own colour (Okabe-Ito, so the pair stays
# distinguishable in greyscale and to colourblind readers). PPO holds several runs rather
# than one design front, so it is left out.
ALGO_COLORS = {"MDH": "#0072b2", "MLP": "#d55e00"}


def load_front(algo: str, run) -> codesign.Grid | None:
    """The run's front grid, or None (with a warning) when there is nothing to read.

    ``FINAL_CONFIGS`` carries an :class:`icra.Run` only for the algorithms already
    searched; the others hold a placeholder that names no dataset.
    """
    if not isinstance(run, icra.Run):
        print(f"warning: {algo} has no run configured; skipping")
        return None
    path = Path(run.dataset_path) / run.run_id / FRONT_NAME
    if not path.exists():
        print(f"warning: {algo} has no dataset at {path}; skipping")
        return None
    return codesign.Grid.load(path)


def plot_front(ax, grid: codesign.Grid, objs, color: str, label: str, labels) -> None:
    """One objective pair's projection of a front: its non-dominated points, joined."""
    rewards = grid.mean_rewards.reshape(-1, grid.n_r)[:, objs]
    mop.plot_pareto(
        ax                   = ax,
        pareto               = rewards,
        colors               = color,
        objective            = [labels[o] for o in objs],
        show_dominated       = False,
        nondominated_s       = 30,
        outline_nondominated = 1.0,
        set_lims             = False,
        marker               = "o",
    )
    # A projection's own frontier, sorted along the x objective so the join is monotone.
    front = rewards[get_nondominated(rewards)]
    ax.plot(*front[np.argsort(front[:, 0])].T, color=color, lw=1.5, zorder=0, label=label)


def main(args) -> None:
    out_dir = Path(args.out_dir)
    fronts = {
        algo: grid
        for algo, run in icra.FINAL_CONFIGS[args.robot].items()
        if algo in ALGO_COLORS
        and (grid := load_front(algo, run)) is not None
    }
    if not fronts:
        print(f"no fronts found for {args.robot}; nothing to plot")
        return

    grid = next(iter(fronts.values()))
    labels = [label.capitalize() for label in codesign.objective_labels(grid.objectives)]

    for objs in combinations(range(grid.n_r), 2):
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
        for algo, front in fronts.items():
            plot_front(ax, front, objs, ALGO_COLORS[algo], algo, labels)
        ax.legend(fontsize=10, frameon=False)
        codesign.dress_axis(ax)
        name = "_".join(labels[o].lower() for o in objs)
        fig.savefig(out_dir / f"subopt_{args.robot}_{name}.{args.format}")
        plt.close(fig)

    print(f"wrote {' vs '.join(fronts)} figures to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=str, default="cheetah",
                        choices=sorted(icra.FINAL_CONFIGS),
                        help="which robot's runs to overlay the fronts of")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                        help="directory the figures are written into; the datasets are "
                             "read from each run's own dataset_path")
    parser.add_argument("--format", type=str, default="svg", choices=("svg", "png", "pdf"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
