import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch
from scipy.stats import binned_statistic, gaussian_kde

import moplayground as mop
import minimal_mjx as mm

from codesign.utils.grid import Grid
import colorstamps

# TODO: ensure docstrings describe all arguments for all functions
# TODO: delete any unused functions

def design_colors(n_designs: int, cmap: str = "viridis") -> np.ndarray:
    """One distinct colour per design."""
    return plt.get_cmap(cmap)(np.linspace(0.0, 1.0, n_designs))

def get_colors(arr: np.ndarray, cmap: str = 'viridis') -> np.ndarray:
    """Makes a list of colors (N, 3) from arr (N, M), where N is the number
    of points and M is the number of elements per item in arr."""
    arr = np.atleast_2d(arr)

    if arr.shape[1] == 1:
        return plt.get_cmap(name=cmap)(arr)

    elif arr.shape[1] == 2:
        colors, _ = colorstamps.apply_stamp(arr[:, 0], arr[:, 1], 'peak')
        return colors
    elif arr.shape[1] == 3:
        colors = arr.astype(float, copy=True)
        colors -= colors.min(axis=0)
        spans = colors.max(axis=0)
        spans[spans == 0] = 1
        colors /= spans
        return colors

    raise NotImplementedError(f'get_colors expects (N, 1), (N, 2), or (N, 3); got {arr.shape}')


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
    grids: list,                   # list of Grid
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


INK, MUTED = "#0b0b0b", "#52514e"   # chart ink and recessive furniture


def dress_axis(ax: plt.Axes) -> plt.Axes:
    """Apply the house style: recessive grid, muted ticks, no top/right spines."""
    ax.grid(True, color=INK, alpha=0.12, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    return ax


DENSITY_RESOLUTION = 200  


def predictor_density(designs: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """Gaussian-KDE density of a design predictor's samples, evaluated at ``points``.

    Args:
        designs: ``(n_samples, design_dim)`` designs drawn from ``f(. | w)``.
        points: ``(design_dim, n_points)`` query locations, or ``(n_points,)`` in 1-D.

    Returns ``(n_points,)`` densities, or ``None`` when the sample cannot support a KDE
    (under two samples, or no spread along an axis, both of which make its bandwidth
    covariance singular).
    """
    dataset = np.atleast_2d(np.asarray(designs).T)
    if dataset.shape[1] < 2 or np.any(dataset.std(axis=1) < 1e-12):
        return None
    return gaussian_kde(dataset)(points)



def _sweep_curve(
    designs: np.ndarray,
    returns: np.ndarray,
    bins: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard deviation of ``returns`` along the design axis.

    With ``bins`` set, every return is pooled into one of that many bins spanning the
    design axis, so the spread is over the designs (and repetitions) sharing a bin.
    Otherwise each design keeps its own position and the spread is over its repetitions.

    Args:
        designs: ``(n_designs, 1)`` swept designs.
        returns: ``(n_designs,)`` or ``(n_designs, n_reps)`` scalarized returns.
        bins: number of bins along the design axis, or ``None`` to keep each design.

    Returns ``(x, mean, std)``, ascending in ``x``; empty bins are dropped.
    """
    x, returns = np.asarray(designs)[:, 0], np.asarray(returns)
    if bins is None:
        order = np.argsort(x)
        if returns.ndim == 1:
            return x[order], returns[order], np.zeros_like(returns, dtype=float)
        return x[order], returns[order].mean(axis=1), returns[order].std(axis=1)

    # Each repetition is its own sample, sitting at its design's position.
    reps = returns.shape[1] if returns.ndim == 2 else 1
    xs, values = np.repeat(x, reps), returns.reshape(-1)
    mean, edges, _ = binned_statistic(xs, values, statistic="mean", bins=bins)
    std, _, _ = binned_statistic(xs, values, statistic="std", bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    drawable = ~np.isnan(mean)  # bins no design landed in
    return centers[drawable], mean[drawable], std[drawable]


def plot_design_sweep_1d(
    ax                  : plt.Axes,
    designs             : np.ndarray,           # (n_designs, 1)
    returns             : np.ndarray,           # (n_designs,) or (n_designs, n_reps)
    style               : str = "line",
    bins                : int | None = None,
    points              : bool = True,
    sweep_color         : str = 'C0',
    best_point_color    : str = 'C0',
    sweep_label         : str = 'Design sweep',
    band_label          : str = r'Sweep $\pm 1 \sigma$',
    points_label        : str = 'Design rollouts',
    optimum_label       : str = 'Optimal design',

) -> int:
    """Scalarized return against the single design parameter, marking the sweep optimum.

    Args:
        ax: axis to draw on.
        designs: the swept designs; only the first (single) design axis is read.
        returns: each design's scalarized return ``w . R``, optionally with a trailing
            repetition axis (the grid's ``per_cell``).
        style: ``line`` for the returns themselves, ``band`` for a mean line with a
            +/-1 standard deviation region shaded around it.
        bins: pool the designs into this many bins along the design axis, so the curve
            is a binned mean; ``None`` keeps every design at its own position.
        points: scatter the individual returns under a ``band``, so the band's meaning
            is readable off the plot. Ignored by ``line``, which already draws them.
        sweep_color: colour of the sweep line and its shaded region.
        best_point_color: colour of the optimum marker.

    Returns the index of the best-returning design. The optimum marks the best design
    actually swept, not the peak of the (smoothed) curve, so under ``bins`` it can sit
    off the line.
    """
    if style not in ("line", "band"):
        raise ValueError(f"style must be 'line' or 'band', got {style!r}")
    designs, returns = np.asarray(designs), np.asarray(returns)
    if style == "band" and bins is None and returns.ndim == 1:
        raise ValueError(
            "style='band' needs a spread to shade: pass returns with a repetition axis, "
            "shape (n_designs, n_reps), or set bins to pool along the design axis."
        )

    if style == "band" and points:
        reps = returns.shape[1] if returns.ndim == 2 else 1
        ax.plot(
            np.repeat(designs[:, 0], reps),
            returns.reshape(-1),
            ".",
            ms     = 2,
            color  = MUTED,
            alpha  = 0.35,
            zorder = 1,
            label  = points_label,
        )

    x, mean, std = _sweep_curve(designs, returns, bins)
    if style == "band":
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color  = sweep_color,
            alpha  = 0.25,
            lw     = 0,
            zorder = 2,
            label  = band_label,
        )
    ax.plot(
        x,
        mean,
        "-o" if style == "line" else "-",
        ms     = 3,
        lw     = 1.5 if style == "line" else 2,
        color  = sweep_color,
        zorder = 3,
        label  = sweep_label,
    )

    per_design = returns if returns.ndim == 1 else returns.mean(axis=1)
    best = int(np.argmax(per_design))
    ax.plot(
        designs[best, 0],
        per_design[best],
        "*",
        ms     = 15,
        color  = best_point_color,
        zorder = 4,
        label  = optimum_label,
    )
    ax.set_xlabel("design $d$", color=MUTED, fontsize=9)
    ax.set_ylabel("scalarized return $w \\cdot R$", color=MUTED, fontsize=9)
    dress_axis(ax)
    return best


def plot_design_predictor(
    ax                    : plt.Axes,
    designs               : np.ndarray,           # (n_samples, 1)
    mode                  : np.ndarray,              # (1,) or scalar
    density_resolution    : int = DENSITY_RESOLUTION,
    density_color         = 'C3',
    colormap_name         = 'design_predictor'
) -> bool:
    """Overlay a 1-D design predictor on ``ax``: its mean design, and how densely it
    samples the design axis.

    The density is shaded over the full height of the axis, so it reads as a marginal
    over the design rather than as a second curve sharing the return axis.

    Args:
        ax: axis whose x-axis is the design parameter, e.g. from
            :func:`plot_design_sweep_1d`.
        designs: designs drawn from ``f(. | w)``; only the first design axis is read.
        mode: the design the predictor centres on, drawn as a vertical line.
        density_resolution: points at which the KDE is evaluated across the x-axis.

    Returns whether the density was drawable; see :func:`predictor_density`.
    """
    ax.axvline(
        np.reshape(mode, -1)[0],
        ls    = "--",
        lw    = 2,
        color = density_color,
        label = "design predictor $f(w)$ mean",
    )

    # imshow rescales the axes, so pin the limits the sweep established.
    extent = (*ax.get_xlim(), *ax.get_ylim())
    density = predictor_density(designs, np.linspace(*extent[:2], density_resolution))
    if density is None:
        return False
    
    cmap = LinearSegmentedColormap.from_list(
        colormap_name, 
        [to_rgba(density_color, 0.0), to_rgba(density_color, 0.4)]
    )
    ax.imshow(
        density[None],
        extent        = extent,
        aspect        = "auto",
        origin        = "lower",
        interpolation = "bilinear",
        cmap          = cmap,
        vmin          = 0,
        zorder        = 0,
    )
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    return True


def plot_pareto_statistics(
    iterations: list,
    hvs: np.ndarray,
    sps: np.ndarray,
    hv_label: str = "Hypervolume (averaged over designs)",
):
    """Hypervolume and front spacing vs. env steps, one panel each.

    ``hvs`` and ``sps`` is per-eval hypervolume and spacing, respectively;
    ``hv_label`` names how they were pooled over the design axis.
    """
    fig, (hv_ax, sp_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 5.5), height_ratios=[1, 1]
    )
    panels = (
        (hv_ax, hvs, "#2a78d6", hv_label),
        (sp_ax, sps, "#eb6834", "Front spacing"),
    )
    for ax, values, color, label in panels:
        ax.plot(iterations, values, color=color, lw=2, marker="o", ms=4.5)
        ax.set_ylabel(label, color=INK)
        dress_axis(ax)

    last = f"HV {hvs[-1]:.3g}   spacing {sps[-1]:.3g}"
    hv_ax.set_title(f"Pareto front progress   ({last})", loc="left", fontsize=11)
    sp_ax.set_xlabel("environment steps")
    fig.tight_layout()
    return fig


@dataclass(frozen=False)
class MODesignTrainingPlottingInfo:
    """
    Practical class for holding plotting/evaluation info during training. 
    
    Aux holds one extra value per eval -- a quantity derived from the grids, or a second
    grid the run evaluates -- so its lists stay aligned with ``iterations``.
    """
    start_time    : float
    iterations    : list = field(default_factory=list)
    grids         : list = field(default_factory=list)
    times         : list = field(default_factory=list)
    labels        : list = field(default_factory=list)
    aux           : dict[str, list] = field(default_factory=dict)
    ref_point     : list | None = None  # hypervolume reference; None means the origin

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


def load_training_data(
    training_data: MODesignTrainingPlottingInfo,
    save_dir: Path,
    csv_name: str,
    aux_grids: dict[str, str] | None = None,
    aux_fn=None,
    before: int | None = None,
) -> MODesignTrainingPlottingInfo:
    """Refill ``training_data`` from the evals in a previous run.

    Auxillary (aux) variables are not always stored, but are rather computed
    from the saved grids. This function recomputes them so that the history
    is represented in a resumed run.

    ``before`` is the step the run restarts from: evals at or past it are left out, the
    continued run evaluating that state again as its own first entry.
    """
    save_dir = Path(save_dir)
    path = save_dir / csv_name
    if not path.exists():
        return training_data
    frame = pd.read_csv(path)
    if before is not None:
        frame = frame[frame["iters"] < before]
    if frame.empty:
        return training_data
    # A run that never wrote a second grid has none to read back for any of its steps.
    aux_grids = {
        key: prefix for key, prefix in (aux_grids or {}).items()
        if (save_dir / f"{prefix}_{frame['iters'].iloc[0]}.npz").exists()
    }
    for step, when in zip(frame["iters"], frame["times"]):
        grid = Grid.load(save_dir / f"eval_grid_{step}.npz")
        training_data.update(
            num_steps = int(step),
            grid      = grid,
            time      = float(when),
            **{k: Grid.load(save_dir / f"{p}_{step}.npz") for k, p in aux_grids.items()},
            **(aux_fn(grid) if aux_fn else {}),
        )
    return training_data


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
    """Log Pareto statistics per design of the eval grid.

    """
    print_training_update(num_steps)
    times.append(time.time())
    grid = metrics['eval_grid']
    mean_rewards = grid.mean_rewards
    per_design = [
        mop.get_pareto_statistics(mean_rewards[d], ref_point=training_data.ref_point)
        for d in range(mean_rewards.shape[0])
    ]
    hv    = float(np.mean([h for h, _ in per_design]))
    sp    = float(np.mean([s for _, s in per_design]))
    sp_sd = float(np.std([s for _, s in per_design]))
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
                'Mean Hypervolume'       : hv,
                'Mean Front Spacing'     : sp,
                'Front Spacing Std. Dev.': sp_sd,
                **scalar_metrics(metrics),
            },
            step=num_steps,
        )

    if save_dir:
        grid.save(save_dir / f"eval_grid_{num_steps}.npz")
        fig = plot_pareto_statistics(training_data.iterations, hvs, sps)
        fig.savefig(save_dir / 'progress.svg')
        plt.close(fig)


def plot_design_pareto_progress(
    num_steps: int,
    metrics: dict,
    training_data: MODesignTrainingPlottingInfo,
    times: list,
    save_dir: Path = None,
    run: wandb.Run = None,
    **kwargs
):
    """Log Design-Pareto statistics. i.e. only Pareto-optimal designs are 
    included in the front.
    """
    print_training_update(num_steps)
    times.append(time.time())
    grid = metrics['eval_grid']
    mean_rewards = grid.mean_rewards  # (n_designs, n_tradeoffs, num_objectives)
    hv, sp = mop.get_pareto_statistics(
        mean_rewards.reshape(-1, mean_rewards.shape[-1]),
        ref_point = training_data.ref_point,
    )
    training_data.update(
        num_steps = num_steps,
        grid      = grid,
        time      = time.time(),
        hv        = float(hv),
        sp        = float(sp),
    )

    hvs = np.asarray(training_data.aux['hv'])
    sps = np.asarray(training_data.aux['sp'])

    if run:
        run.log(
            {
                'Pareto-Optimal Design Hypervolume'   : float(hv),
                'Pareto-Optimal Design Spacing'       : float(sp),
                **scalar_metrics(metrics),
            },
            step=num_steps,
        )

    if save_dir:
        training_data.save(save_dir / 'design_pareto_progress.csv')
        grid.save(save_dir / f"eval_grid_{num_steps}.npz")
        fig = plot_pareto_statistics(
            training_data.iterations, hvs, sps,
            hv_label = "Hypervolume (pooled over designs)",
        )
        fig.savefig(save_dir / 'progress.svg')
        plt.close(fig)


def plot_design_rewards(ax: plt.Axes, grid, colors=None) -> plt.Axes:
    """Every rollout's return at its design, with each design's mean marked.

    The x axis is the design itself when it is one-dimensional and the design's index
    otherwise, there being no single axis to place a multi-dimensional design on.
    """
    returns = grid.scalarized_rewards.reshape(grid.n_designs, -1)  # (M, K * C)
    designs = np.asarray(grid.designs)[:, 0]                       # (M, design_dim)
    colors = design_colors(grid.n_designs) if colors is None else colors
    one_d = designs.shape[-1] == 1
    x = designs[:, 0] if one_d else np.arange(grid.n_designs)

    for i, (xi, row) in enumerate(zip(x, returns)):
        ax.plot(
            np.full(row.size, xi), row, ".",
            ms=4, alpha=0.35, color=colors[i], zorder=1,
        )
        ax.plot(xi, row.mean(), "*", ms=11, color=colors[i], zorder=2)
    ax.plot([], [], ".", ms=4, color=MUTED, label="rollout")
    ax.plot([], [], "*", ms=11, color=MUTED, label="design mean")
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    ax.set_xlabel("design $d$" if one_d else "design index", color=MUTED, fontsize=9)
    ax.set_ylabel("scalarized return $w \\cdot R$", color=MUTED, fontsize=9)
    return dress_axis(ax)


def plot_design_learning_curves(ax: plt.Axes, iterations, grids, colors=None) -> plt.Axes:
    """Each design's mean return against the training step, one line per design."""
    means = np.stack([
        g.scalarized_rewards.reshape(g.n_designs, -1).mean(axis=1) for g in grids
    ])  # (n_evals, M)
    colors = design_colors(means.shape[1]) if colors is None else colors
    for i, column in enumerate(means.T):
        ax.plot(iterations, column, "-", lw=1.2, color=colors[i], zorder=2)
    ax.set_xlabel("environment steps", color=MUTED, fontsize=9)
    ax.set_ylabel("mean return", color=MUTED, fontsize=9)
    return dress_axis(ax)


def save_designs(designs, path: Path) -> None:
    """One design per row, indexed to match a grid's design axis."""
    pd.DataFrame(
        designs, columns=[f"d{i}" for i in range(np.shape(designs)[-1])],
    ).rename_axis("index").to_csv(path)


def save_design_rewards_figure(
    path: Path,
    num_steps: int,
    iterations: list,
    grids: list,
    label: str = None,
    run: wandb.Run = None,
    log_key: str = None,
) -> None:
    """The latest grid's per-design returns beside every design's learning curve.

    ``label`` names the design set, for a run that plots more than one.
    """
    name = "per-design" if label is None else f"{label} per-design"
    fig, (latest, curves) = plt.subplots(1, 2, figsize=(11, 4))
    plot_design_rewards(latest, grids[-1])
    plot_design_learning_curves(curves, iterations, grids)
    latest.set_title(f"{name} return, step {num_steps}", color=INK, fontsize=10)
    curves.set_title(f"{name} mean return", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    if run and log_key:
        run.log({log_key: wandb.Html(path.read_text())}, step=num_steps)


def plot_design_rewards_progress(
    num_steps: int,
    metrics: dict,
    training_data: MODesignTrainingPlottingInfo,
    times: list,
    save_dir: Path = None,
    run: wandb.Run = None,
    **kwargs,
):
    """Per-design evaluation, for algos whose designs each carry their own policy.

    A pooled mean cannot say whether every design is training, so this draws the eval's
    individual rollout returns against their design beside each design's learning curve.

    An anchored ``design_hypernetwork`` run also evaluates the designs it trains on.
    Those are a different design set from ``eval_grid``, so index ``i`` means a different
    robot in each, and they get their own figure, npz and designs.csv rather than sharing
    the held-out grid's.
    """
    print_training_update(num_steps)
    times.append(time.time())
    grid = metrics["eval_grid"]
    anchor_grid = metrics.get("anchor_eval_grid")
    training_data.update(
        num_steps=num_steps, grid=grid, time=time.time(),
        **({} if anchor_grid is None else {"anchor_grid": anchor_grid}),
    )

    if run:
        run.log(scalar_metrics(metrics), step=num_steps)

    if save_dir:
        training_data.save(save_dir / "design_rewards_progress.csv")
        grid.save(save_dir / f"eval_grid_{num_steps}.npz")

        # save the designs
        train_designs = metrics.get("train_designs")
        eval_designs = np.asarray(grid.designs)[:, 0]
        save_designs(
            eval_designs if train_designs is None else train_designs,
            save_dir / "designs.csv",
        )
        # A run with only one design set needs no disambiguating label.
        save_design_rewards_figure(
            save_dir / "progress.svg", num_steps, training_data.iterations,
            training_data.grids, None if anchor_grid is None else "held-out",
            run, "design_rewards",
        )
        if anchor_grid is not None:
            # designs.csv holds the anchors, so the held-out grid needs its own index.
            save_designs(eval_designs, save_dir / "eval_designs.csv")
            anchor_grid.save(save_dir / f"anchor_eval_grid_{num_steps}.npz")
            save_design_rewards_figure(
                save_dir / "progress_anchors.svg", num_steps,
                training_data.iterations, training_data.aux["anchor_grid"],
                "anchor", run, "anchor_design_rewards",
            )
