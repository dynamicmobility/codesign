"""Plot where the design predictor points against what the universal policy achieves.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import moplayground as mop
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import codesign

DATA_PATH   = "results/datasets"
OUTPUT_PATH = "scripts/outputs"
SWEEP_NAME     = "sweep.npz"
PREDICTOR_NAME = "predictor.npz"

N_TRADEOFFS = 4      # tradeoffs drawn under --tradeoff_sample random
MAX_COLS    = 3      # subplots per row
DENSITY_LABEL = "$f(d \\mid w)$ sample density"
UP_COLOR, DP_COLOR, BEST_COLOR = "C0", "C3", "C2"

# TODO use the config parameter in the grid to pull what the design predictor mean was


def density_legend_handle() -> Patch:
    """Legend proxy for the shaded density, which is an image/contour set and so is not
    picked up by ``get_legend_handles_labels``."""
    return Patch(color=to_rgba(DP_COLOR, 0.45), label=DENSITY_LABEL)

def select_tradeoffs(tradeoffs, sample, n_tradeoffs, fixed, rng) -> np.ndarray:
    """Indices of the sweep grid's tradeoffs to plot."""
    if fixed is not None:
        return mop.closest_points(tradeoffs, mop.project_to_simplex(fixed))
    if sample == "corners":
        return mop.corner_tradeoffs(tradeoffs)
    return rng.choice(len(tradeoffs), size=min(n_tradeoffs, len(tradeoffs)), replace=False)


def check_shared_tradeoffs(sweep, predictor) -> None:
    """Both grids must sit on the same tradeoffs for a column-by-column comparison."""
    if sweep.tradeoffs.shape != predictor.tradeoffs.shape or not np.allclose(
        sweep.tradeoffs, predictor.tradeoffs, atol=1e-6
    ):
        raise ValueError(
            f"'{SWEEP_NAME}' and '{PREDICTOR_NAME}' were rolled out on different tradeoffs "
            f"{sweep.tradeoffs.shape} vs {predictor.tradeoffs.shape}; regenerate both with "
            f"the same --seed, --n_tradeoffs and env."
        )


def designs_at(dataset, t: int) -> np.ndarray:
    """The designs behind tradeoff column ``t``, returns ``(n_designs, design_dim)``.

    The sweep shares one design set across all tradeoffs; the predictor draws its own per
    tradeoff, so its ``designs`` carry an extra tradeoff axis.
    """
    designs = np.asarray(dataset.designs)
    return designs[:, t] if designs.ndim == 3 else designs


def scalarized(dataset, t: int) -> np.ndarray:
    """``w . R`` per design: the scalar that tradeoff ``w`` asks the design to maximize."""
    return dataset.mean_rewards[:, t] @ np.asarray(dataset.tradeoffs)[t]


def plot_tradeoff_1d(ax, up_designs, up_returns, dp_designs) -> bool:
    """One tradeoff's sweep with the predictor's proposal for it overlaid.

    Returns whether the predictor's sample density was drawable.
    """
    codesign.plot_design_sweep_1d(ax, up_designs, up_returns, style='band', bins=32, points=True)
    return codesign.plot_design_predictor(ax, dp_designs, dp_designs.mean(axis=0))


def plot_tradeoff_2d(ax, up_designs, up_returns, dp_designs) -> bool:
    """As :func:`plot_tradeoff_1d`, as a surface over the two design parameters.

    The sweep designs are a space-filling (Sobol) sample rather than a mesh, so the
    surface is the Delaunay triangulation of those scattered points.
    """
    # TODO: fix this.
    # Kept translucent so the markers below the surface still read through it.
    ax.plot_trisurf(up_designs[:, 0], up_designs[:, 1], up_returns, cmap="viridis",
                    alpha=0.55, linewidth=0, label="universal policy sweep")

    best = np.argmax(up_returns)
    ax.scatter(*up_designs[best], up_returns[best], marker="*", s=180, color=BEST_COLOR,
               depthshade=False, label="sweep optimum")

    span = (up_returns.min(), up_returns.max())

    proposal = dp_designs.mean(axis=0)
    ax.plot([proposal[0]] * 2, [proposal[1]] * 2, span,
            ls="--", lw=2, color=DP_COLOR, label="design predictor $f(w)$ mean")

    # The 2-D density goes on the floor of the box, where it reads as a contour map of
    # the design plane rather than fighting the surface for the same space.
    axis_grid = [np.linspace(min(up_designs[:, i].min(), dp_designs[:, i].min()),
                             max(up_designs[:, i].max(), dp_designs[:, i].max()),
                             DENSITY_RESOLUTION) for i in (0, 1)]
    xx, yy = np.meshgrid(*axis_grid)
    density = predictor_density(dp_designs, np.vstack([xx.ravel(), yy.ravel()]))
    if density is not None:
        ax.contourf(xx, yy, density.reshape(xx.shape), zdir="z", offset=span[0],
                    levels=np.linspace(0, density.max(), 12), cmap=DENSITY_CMAP)
        ax.set_zlim(span)
    ax.set_xlabel("$d_0$")
    ax.set_ylabel("$d_1$")
    ax.set_zlabel("$w \\cdot R$")
    return density is not None


def tradeoff_title(tradeoff, labels) -> str:
    """``w`` written out against the objective it weights."""
    if labels is None:
        return "w = " + np.array2string(tradeoff, precision=2)
    return ", ".join(f"{name} {value:.2f}" for name, value in zip(labels, tradeoff))


def main(args) -> None:
    data_path = Path(args.data_path)
    sweep     = codesign.DesignTradeoffDataset.load(data_path / SWEEP_NAME)
    predictor = codesign.DesignTradeoffDataset.load(data_path / PREDICTOR_NAME)
    check_shared_tradeoffs(sweep, predictor)

    design_dim = np.asarray(sweep.designs).shape[-1]
    if design_dim not in (1, 2):
        raise ValueError(
            f"designs are {design_dim}-D; this figure only covers 1-D (line) and 2-D "
            f"(surface) design spaces."
        )

    indices = select_tradeoffs(
        np.asarray(sweep.tradeoffs), args.tradeoff_sample, args.n_tradeoffs,
        args.fixed_tradeoff, np.random.default_rng(args.seed),
    )
    labels = codesign.objective_labels(sweep.objectives)

    ncols = min(len(indices), MAX_COLS)
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False,
        subplot_kw={"projection": "3d"} if design_dim == 2 else None,
        layout="constrained",
    )
    plot_fn = plot_tradeoff_1d if design_dim == 1 else plot_tradeoff_2d

    drew_density = False
    for ax, t in zip(axes.ravel(), indices):
        drew_density |= plot_fn(
            ax, designs_at(sweep, t), scalarized(sweep, t), designs_at(predictor, t)
        )
        ax.set_title(tradeoff_title(np.asarray(sweep.tradeoffs)[t], labels), fontsize=10)
    for ax in axes.ravel()[len(indices):]:
        ax.set_visible(False)

    # One shared legend below the grid; constrained layout reserves the room for it.
    handles, handle_labels = axes.ravel()[0].get_legend_handles_labels()
    if drew_density:
        handles.append(density_legend_handle())
        handle_labels.append(DENSITY_LABEL)
    fig.legend(handles, handle_labels, loc="outside lower center",
               ncol=min(len(handles), 2 * ncols), frameon=False)
    fig.suptitle("Design predictor proposal vs. universal policy sweep")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / "dp_vs_up.svg")
    plt.close(fig)
    print(f"{len(indices)} tradeoffs -> {output_path / 'dp_vs_up.svg'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=str, default=DATA_PATH,
                        help=f"directory holding {SWEEP_NAME} and {PREDICTOR_NAME}")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH,
                        help="directory the svg is written into")
    parser.add_argument("--tradeoff_sample", type=str, default="corners",
                        choices=("corners", "random"),
                        help="which of the grid's tradeoffs to plot")
    parser.add_argument("--fixed_tradeoff", type=float, nargs="+", default=None,
                        help="plot the grid tradeoff nearest this vector, which is first "
                             "projected onto the simplex; overrides --tradeoff_sample")
    parser.add_argument("--n_tradeoffs", type=int, default=N_TRADEOFFS,
                        help="how many tradeoffs to draw under --tradeoff_sample random")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for --tradeoff_sample random")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
