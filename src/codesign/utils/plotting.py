import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

import moplayground as mop
import minimal_mjx as mm


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
        mop.plot_pareto(
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
    colored by the design that achieves it."""
    rewards = np.asarray(rewards)
    n_designs, n_tradeoffs, num_obj = rewards.shape
    rewards = rewards.reshape(-1, num_obj)
    # objectives = objective_labels(objectives)
    if colors is None:
        colors = design_colors(n_designs)
        colors = np.repeat(colors, n_tradeoffs, axis=0)

    mop.plot_pareto(
        ax,
        pareto=rewards,
        colors=colors,
        objective=objectives,
        dominated_alpha=0.15 if show_dominated else 0.0,
    )

    return ax


def plot_sequential_design_paretos(
    ax_titles: list,
    grids: list,                   # list of DesignTradeoffRolloutGrid
):
    """One subplot per checkpoint, each showing every design's Pareto frontier."""
    rewards_seq = [g.mean_rewards for g in grids]
    n_designs, _, num_obj = rewards_seq[0].shape
    colors = design_colors(n_designs)

    nrows, ncols = mm.get_subplot_grid(len(ax_titles))
    subplot_kw = {"projection": "3d"} if num_obj == 3 else {}
    fig, axs = plt.subplots(nrows, ncols, subplot_kw=subplot_kw, squeeze=False)
    axs = axs.flatten()

    for ax, title, grid, rewards in zip(axs, ax_titles, grids, rewards_seq):
        plot_design_paretos(ax, rewards, grid.objectives, grid.designs, colors)
        ax.set_title(title)
    for ax in axs[len(ax_titles):]:
        ax.axis("off")

    handles, labels = axs[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.set_size_inches((4 * ncols, 4 * nrows))
    fig.tight_layout()
    return fig, axs


def plot_pareto_statistics(
    iterations: list,
    hvs: np.ndarray,
    sps: np.ndarray,
):
    """Hypervolume and front spacing vs. env steps, one panel each.

    ``hvs`` and ``sps`` is per-eval hypervolume and spacing averaged over 
    designs, respectively.    
    """
    fig, (hv_ax, sp_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 5.5), height_ratios=[1, 1]
    )
    panels = (
        (hv_ax, hvs, "#2a78d6", "Hypervolume (averaged over designs)"),
        (sp_ax, sps, "#eb6834", "Front spacing"),
    )
    for ax, values, color, label in panels:
        ax.plot(iterations, values, color=color, lw=2, marker="o", ms=4.5)
        ax.set_ylabel(label, color="#0b0b0b")
        ax.grid(True, color="#0b0b0b", alpha=0.12, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#52514e")
        ax.tick_params(colors="#52514e")

    last = f"HV {hvs[-1]:.3g}   spacing {sps[-1]:.3g}"
    hv_ax.set_title(f"Pareto front progress   ({last})", loc="left", fontsize=11)
    sp_ax.set_xlabel("environment steps")
    fig.tight_layout()
    return fig


@dataclass(frozen=False)
class MODesignTrainingPlottingInfo:
    """
    Practical class for holding plotting/evaluation info during training. 
    
    Aux should only contain data that can be computed from class attributes but 
    may be convenient to hold on to.
    """
    start_time    : float
    iterations    : list = field(default_factory=list)
    grids         : list = field(default_factory=list)
    times         : list = field(default_factory=list)
    labels        : list = field(default_factory=list)
    aux           : dict[str, list] = field(default_factory=dict)

    def save(self, save_dir):
        pd.DataFrame(
            {"times": self.times, "iters": self.iterations}
        ).to_csv(save_dir)

    def update(self, num_steps, grid, time, **aux_kwargs):
        self.iterations.append(num_steps)
        self.grids.append(grid)
        self.times.append(time)
        for key, value in aux_kwargs.items():
            self.aux.setdefault(key, []).append(value)


def scalar_metrics(metrics: dict) -> dict:
    """The numeric-scalar entries of ``metrics``, as floats."""
    return {
        k: float(v) for k, v in metrics.items()
        if isinstance(v, (int, float, np.number))
    }


def print_training_update(num_steps):
    print('=== TRAINING EPOCH ===')
    print('time',datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z") )
    print('num_steps', num_steps)


def plot_mo_design_progress(
    num_steps: int,
    metrics: dict,
    training_data: MODesignTrainingPlottingInfo,
    save_dir: Path,
    run: wandb.Run = None,
    times = [],
    **kwargs,
):
    print_training_update(num_steps)
    grid = metrics['eval_grid']
    training_data.update(num_steps=num_steps, grid=grid, time=time.time())
    training_data.save(save_dir / "mo_design_progress.csv")
    grid.save(save_dir / f"eval_grid_{num_steps}.npz")

    times.append(datetime.now())
    fig, _ = plot_sequential_design_paretos(
        ax_titles = training_data.iterations,
        grids     = training_data.grids,
    )
    fig.savefig(save_dir / "mo_design_progress.svg")
    plt.close(fig)

    if run:
        with open(save_dir / "mo_design_progress.svg", "r") as f:
            svg = f.read()
        run.log({"pareto_plot": wandb.Html(svg)}, step=num_steps)
        run.log(scalar_metrics(metrics), step=num_steps)
        
def plot_mean_hv_progress(
    num_steps: int,
    metrics: dict,
    training_data: MODesignTrainingPlottingInfo,
    times: list,
    save_dir: Path = None,
    run: wandb.Run = None,
    **kwargs
):
    """Log Pareto statistics accumulated over the *design* axis of the eval grid.

    Hypervolume is summed over designs. Spacing is averaged over them.
    """
    print_training_update(num_steps)
    times.append(time.time())
    grid = metrics['eval_grid']
    mean_rewards = grid.mean_rewards
    per_design = [
        mop.get_pareto_statistics(mean_rewards[d])
        for d in range(mean_rewards.shape[0])
    ]
    hv = float(np.mean([h for h, _ in per_design]))
    sp = float(np.mean([s for _, s in per_design]))
    training_data.update(
        num_steps = num_steps,
        grid      = grid,
        time      = time.time(),
        hv        = hv,
        sp        = sp,
    )

    hvs = np.asarray(training_data.aux['hv'])
    sps = np.asarray(training_data.aux['sp'])

    if run:
        run.log(
            {
                'Mean Hypervolume' : hv,
                'Average Spacing'        : float(sps.mean()),
                'Spacing Standard Dev.'  : float(sps.std()),
                **scalar_metrics(metrics),
            },
            step=num_steps,
        )

    if save_dir:
        grid.save(save_dir / f"eval_grid_{num_steps}.npz")
        fig = plot_pareto_statistics(training_data.iterations, hvs, sps)
        fig.savefig(save_dir / 'progress.svg')
        plt.close(fig)

