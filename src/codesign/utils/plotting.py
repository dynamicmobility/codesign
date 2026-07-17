import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from minimal_mjx.utils.plotting import get_subplot_grid
from moplayground.utils.pareto import get_nondominated
from moplayground.utils.plotting import plot_pareto


def design_colors(n_designs: int, cmap: str = "viridis") -> np.ndarray:
    """One distinct colour per design."""
    return plt.get_cmap(cmap)(np.linspace(0.0, 1.0, n_designs))


def objective_labels(objectives) -> list[str] | None:
    """Flatten nested objective names (e.g. ``[['run'], ['energy']]``) to strings."""
    if objectives is None:
        return None
    return [
        "+".join(map(str, o)) if isinstance(o, (list, tuple)) else str(o)
        for o in objectives
    ]


def _design_label(designs, d: int) -> str | None:
    if designs is None:
        return None
    return f"d={np.asarray(designs)[d].reshape(-1)[0]:.2f}"


def plot_design_paretos(
    ax,
    rewards: np.ndarray,           # (num_designs, num_tradeoffs, num_objectives)
    objectives: list = None,
    designs: np.ndarray = None,
    colors: np.ndarray = None,
    show_dominated: bool = True,
):
    """Overlay each design's Pareto frontier on a single axis, coloured by design."""
    rewards = np.asarray(rewards)
    n_designs, _, num_obj = rewards.shape
    objectives = objective_labels(objectives)
    if colors is None:
        colors = design_colors(n_designs)

    for d in range(n_designs):
        pts = rewards[d]  # (num_tradeoffs, num_objectives)
        plot_pareto(
            ax, 
            rewards[d], 
            colors=colors[d], 
            objective=objectives,
            dominated_alpha=0.1 if show_dominated else 0.0, 
            # label=_design_label(designs, d), 
            set_lims=False,
        )
    return ax

def plot_design_objective_pareto(
    ax,
    rewards: np.ndarray,           # (num_designs, num_tradeoffs, num_objectives)
    objectives: list = None,
    designs: np.ndarray = None,
    colors: np.ndarray = None,
    show_dominated = True,
):
    """Global Pareto frontier over all (design x tradeoff) points, each frontier point
    coloured by the design that achieves it."""
    rewards = np.asarray(rewards)
    n_designs, n_tradeoffs, num_obj = rewards.shape
    rewards = rewards.reshape(-1, num_obj)
    # objectives = objective_labels(objectives)
    if colors is None:
        colors = design_colors(n_designs)
        colors = np.repeat(colors, n_tradeoffs, axis=0)

    plot_pareto(
        ax,
        pareto=rewards,
        colors=colors,
        objective=objectives,
        dominated_alpha=0.15 if show_dominated else 0.0,
    )

    return ax


def plot_sequential_design_paretos(
    ax_titles: list,
    rewards_seq: list,             # list of (num_designs, num_tradeoffs, num_objectives)
    objectives: list = None,
    designs: np.ndarray = None,
):
    """One subplot per checkpoint, each showing every design's Pareto frontier."""
    rewards_seq = [np.asarray(r) for r in rewards_seq]
    n_designs, _, num_obj = rewards_seq[0].shape
    colors = design_colors(n_designs)

    nrows, ncols = get_subplot_grid(len(ax_titles))
    subplot_kw = {"projection": "3d"} if num_obj == 3 else {}
    fig, axs = plt.subplots(nrows, ncols, subplot_kw=subplot_kw, squeeze=False)
    axs = axs.flatten()

    for ax, title, rewards in zip(axs, ax_titles, rewards_seq):
        plot_design_paretos(ax, rewards, objectives, designs, colors)
        ax.set_title(title)
    for ax in axs[len(ax_titles):]:
        ax.axis("off")

    handles, labels = axs[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.set_size_inches((4 * ncols, 4 * nrows))
    fig.tight_layout()
    return fig, axs


@dataclass(frozen=False)
class MODesignTrainingPlottingInfo:
    start_time: float
    iterations: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    tradeoffs: list = field(default_factory=list)
    designs: list = field(default_factory=list)
    times: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def save(self, save_dir):
        pd.DataFrame(
            {"times": self.times, "iters": self.iterations}
        ).to_csv(save_dir)


def plot_mo_design_progress(
    num_steps: int,
    metrics: dict,
    training_data: MODesignTrainingPlottingInfo,
    save_dir: Path,
    run: wandb.Run = None,
    **kwargs,
):
    print(datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z"))

    training_data.iterations.append(num_steps)
    training_data.rewards.append(np.asarray(metrics["reward"]))
    training_data.tradeoffs.append(np.asarray(metrics["tradeoffs"]))
    training_data.designs.append(np.asarray(metrics["designs"]))
    training_data.times.append(time.time())
    training_data.save(save_dir / "mo_design_progress.csv")

    fig, _ = plot_sequential_design_paretos(
        ax_titles   = training_data.iterations,
        rewards_seq = training_data.rewards,
        objectives  = training_data.labels,
        designs     = training_data.designs[-1],
    )
    fig.savefig(save_dir / "mo_design_progress.svg")
    plt.close(fig)

    if run:
        with open(save_dir / "mo_design_progress.svg", "r") as f:
            svg = f.read()
        run.log({"pareto_plot": wandb.Html(svg)}, step=num_steps)
        scalars = {
            k: float(v) for k, v in metrics.items()
            if np.ndim(v) == 0 and not isinstance(v, str)
        }
        run.log(scalars, step=num_steps)
