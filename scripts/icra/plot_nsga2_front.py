"""Plot the design x tradeoff Pareto frontier NSGA-II searched out.

Reads the grids ``nsga2_design_front.py`` writes -- one ``(design, tradeoff)`` pair per
cell -- and draws each objective pair's projection of the frontier, every point coloured by
the design that achieves it. A baseline sweep (``generate_data.py``'s ``sweep.npz``) can be
overlaid to show what the search buys over sampling the design box.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated, get_pareto_statistics

import codesign
import colorstamps

OUT_DIR      = "scripts/icra/outputs"
CMAP         = "plasma"
DESIGN_LABEL = "Total link scale"
BASELINE_COLOR = "#9a9a9a"

STAMP_CMAP   = "peak"
STAMP_X_LABEL = "Front Length"
STAMP_Y_LABEL = "Back Length"


def pair_rewards(grid: codesign.Grid) -> np.ndarray:
    """The ``(N, n_r)`` objective vectors of a grid, one row per cell."""
    return grid.mean_rewards.reshape(-1, grid.n_r)


def pair_designs(grid: codesign.Grid) -> np.ndarray:
    """The ``(N, design_dim)`` designs of a grid, one row per cell."""
    return grid.designs.reshape(-1, grid.design_dim)


def color_values(designs: np.ndarray, dims) -> np.ndarray:
    """The ``(N, 1)`` scalar each point is coloured by: its design summed over ``dims``.

    Summing is what makes a multi-dimensional design one number; ``dims`` picks out a
    subset, e.g. ``3 4 5`` for the cheetah's back leg alone.
    """
    dims = range(designs.shape[1]) if dims is None else dims
    return designs[:, list(dims)].sum(axis=1, keepdims=True)


def plot_pair(ax, grid, objs, norm, args, baseline=None):
    """One objective pair's projection of the frontier, with the baseline behind it."""
    labels = codesign.objective_labels(grid.objectives) if args.labels is None else args.labels
    # A 3D front is a surface, so only a 2D projection's points can be joined into a line.
    connect = len(objs) == 2

    if baseline is not None:
        mop.plot_pareto(
            ax             = ax,
            pareto         = pair_rewards(baseline)[:, objs],
            colors         = BASELINE_COLOR,
            objective      = [labels[o] for o in objs],
            show_dominated = args.show_dominated,
            dominated_alpha= 0.15,
            dominated_s    = 6,
            nondominated_s = 18,
            connect        = connect,
            marker         = "s",
            set_lims       = False,
            label          = "design sweep",
        )

    designs = grid.designs.reshape((-1, grid.designs.shape[2]))
    designs_color = np.hstack((np.sum(designs[:, 0:3], axis=1)[:, None], np.sum(designs[:, 3:], axis=1)[:, None]))
    colors, stamp = colorstamps.apply_stamp(designs_color[:, 0], designs_color[:, 1], STAMP_CMAP)

    mop.plot_pareto(
        ax                   = ax,
        pareto               = pair_rewards(grid)[:, objs],
        colors               = colors,
        objective            = [labels[o] for o in objs],
        show_dominated       = args.show_dominated,
        dominated_alpha      = 0.2,
        dominated_s          = 8,
        nondominated_s       = 30,
        outline_nondominated = 1.0,
        connect              = connect,
        marker               = "o",
        set_lims             = False,
        label                = "NSGA-II",
    )
    return ax, stamp


def report(name, grid, ref_point):
    """Print the frontier's size, hypervolume and sparsity, for comparing runs.

    Sparsity is the mean gap between neighbouring frontier points, so a denser frontier
    covering the same region reads as a smaller number.
    """
    rewards = pair_rewards(grid)
    n_front = len(get_nondominated(rewards))
    hypervolume, sparsity = get_pareto_statistics(rewards, ref_point)
    print(
        f"{name:>14}: {n_front:5d} non-dominated of {len(rewards):5d} evaluated, "
        f"hypervolume {hypervolume:12.4f}, sparsity {sparsity:9.5f}"
    )


def main(args) -> None:
    grid = codesign.Grid.load(args.front)
    baseline = None if args.baseline is None else codesign.Grid.load(args.baseline)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_point = None if args.ref_point is None else np.asarray(args.ref_point)
    report("NSGA-II", grid, ref_point)
    if baseline is not None:
        report("design sweep", baseline, ref_point)

    # One colour scale across every figure, spanning both datasets so the two are readable
    # against each other.
    values = color_values(pair_designs(grid), args.color_dims)
    if baseline is not None:
        values = np.vstack([values, color_values(pair_designs(baseline), args.color_dims)])
    norm = Normalize(vmin=values.min(), vmax=values.max())

    labels = codesign.objective_labels(grid.objectives) if args.labels is None else args.labels
    for objs in combinations(range(grid.n_r), 2):
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
        plot_pair(ax, grid, objs, norm, args, baseline)
        ax = codesign.dress_axis(ax)
        # ax.legend(fontsize=8, frameon=False)
        # fig.colorbar(
        #     ScalarMappable(norm=norm, cmap=args.cmap), ax=ax,
        #     label=args.design_label, aspect=50,
        # )
        name = "_".join(labels[o].lower() for o in objs)
        fig.savefig(out_dir / f"nsga2_front_{name}.{args.format}")
        plt.close(fig)

    if grid.n_r == 3:
        fig = plt.figure(figsize=(5.5, 4.5), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        ax, stamp = plot_pair(ax, grid, (0, 1, 2), norm, args, baseline)
        ax = codesign.dress_axis(ax)
        ax.view_init(elev=30, azim=45)
        stamp_ax = stamp.overlay_ax(ax, lower_left_corner=[0.8, 0.15], width=0.2)
        stamp_ax.set_xlabel(STAMP_X_LABEL)
        stamp_ax.set_ylabel(STAMP_Y_LABEL)
        codesign.dress_axis(stamp_ax)
        fig.savefig(out_dir / f"nsga2_front_3d.{args.format}", bbox_inches="tight", pad_inches=0,)
        plt.close(fig)

    print(f"wrote figures to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=str, required=True,
                        help="npz written by nsga2_design_front.py (front or archive)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="npz of a design sweep (generate_data.py) to overlay")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                        help="directory the figures are written into")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="objective names for the axes; defaults to the grid's own")
    parser.add_argument("--design_label", type=str, default=DESIGN_LABEL,
                        help="colorbar label for the design quantity points are coloured by")
    parser.add_argument("--color_dims", type=int, nargs="+", default=None,
                        help="design dimensions summed for the colour; defaults to all "
                             "(e.g. 3 4 5 for the cheetah's back leg alone)")
    parser.add_argument("--ref_point", type=float, nargs="+", default=None,
                        help="hypervolume reference in maximization space; defaults to the "
                             "origin")
    parser.add_argument("--cmap", type=str, default=CMAP)
    parser.add_argument("--format", type=str, default="svg", choices=("svg", "png", "pdf"))
    parser.add_argument("--show_dominated", action="store_true",
                        help="also draw the points the frontier dominates")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
