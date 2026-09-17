"""Call out ``N`` spaced-out ``(design, tradeoff)`` pairs on each 2D slice of a front.

A three-objective front is a surface, and each pair of objectives is one 2D slice of it:
the points that stay non-dominated once the other objectives are projected away. This picks
``--n-callouts`` pairs spread along each of those slices -- concrete designs to render or
quote, rather than the whole frontier -- prints their ``(w, d)``, draws them as stars over
the slice they came from, and writes them out as the arms of a W&B grid sweep, so PPO can
be trained on each pair directly and say what the hypernetwork's front gave up.

The front is the ``nsga3_front.npz`` that ``nsga2_design_front.py`` wrote for the robot's
``MDH`` run, found through that run's ``icra.Run``.

Run from the repository root, where the configured paths resolve:
``python -m scripts.icra.front_callouts --robot cheetah --n-callouts 3``.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import moplayground as mop
from moplayground.utils.pareto import get_nondominated

import codesign

from scripts import icra

ALGO        = "MDH"      # which of the robot's runs the front is read from
FRONT_NAME  = "nsga3_front.npz"
OUT_DIR     = "scripts/icra/outputs"
FORMAT      = "svg"

SWEEP_DIR      = "config/ppo/sweeps"
SWEEP_SAVE_DIR = "results/icra"                 # where the sweep's runs write
SWEEP_METRIC   = "eval/episode_reward.max"      # per-arm best eval, for the sweep table

N_CALLOUTS  = 3          # default pairs called out per 2D slice
COLOR       = "#0072b2"  # the slice's own colour; the callouts are stars of the same
STAR_SIZE   = 260        # callout marker area, in points squared
DECIMALS    = 4          # places the sweep arms are rounded to


def load_front(robot: str) -> codesign.Grid:
    """The robot's ``ALGO`` front grid, from the dataset that run's ``icra.Run`` names."""
    run = icra.FINAL_CONFIGS[robot][ALGO]
    if not isinstance(run, icra.Run):
        raise SystemExit(f"{robot} configures no {ALGO} run to read a front from")
    path = Path(run.dataset_path) / run.run_id / FRONT_NAME
    if not path.exists():
        raise SystemExit(
            f"no dataset at {path}; generate it with "
            f"`python -m scripts.icra.nsga2_design_front --robot {robot} "
            f"--algorithm {ALGO}`"
        )
    return codesign.Grid.load(path)


def spaced_indices(points: np.ndarray, n: int) -> np.ndarray:
    """``n`` indices into ``points``, spread evenly along the frontier they trace.

    The frontier is walked in order of the first objective and parameterized by cumulative
    arc length, then the picks are the points nearest ``n`` equally spaced arc lengths. So
    they are spaced by distance along the curve, not by index -- index spacing would bunch
    them wherever the search happened to sample densely. Each objective is first scaled to
    ``[0, 1]`` across the frontier, so that an objective with a wide range (the cheetah's
    height spans thousands, its energy hundreds) does not set the distance by itself.

    Fewer than ``n`` indices come back when two targets land on the same point.
    """
    order = np.argsort(points[:, 0])
    walk  = points[order]
    span  = np.ptp(walk, axis=0)
    unit  = (walk - walk.min(axis=0)) / np.where(span == 0, 1, span)
    arc   = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(unit, axis=0), axis=1))])
    targets = np.linspace(0, arc[-1], n)
    return order[np.unique(np.abs(arc[:, None] - targets).argmin(axis=0))]


def slice_callouts(rewards: np.ndarray, objs, n: int) -> np.ndarray:
    """Indices into ``rewards`` of the ``n`` pairs called out on the ``objs`` slice.

    The slice is what survives projecting the other objectives away, so a point of the 3D
    front can be dominated here and is not a candidate.
    """
    slice_idx = get_nondominated(rewards[:, objs])
    return slice_idx[spaced_indices(rewards[slice_idx][:, objs], n)]


def report(labels, objs, picks, designs, tradeoffs, rewards) -> None:
    """Print each callout's tradeoff, design and the two objectives it was picked on."""
    print(f"\n{labels[objs[0]]} vs {labels[objs[1]]}")
    for i in picks:
        print(
            f"  w = {np.array2string(tradeoffs[i], precision=3, floatmode='fixed')}"
            f"  d = {np.array2string(designs[i], precision=3, floatmode='fixed')}"
            f"  -> {np.array2string(rewards[i, objs], precision=1, floatmode='fixed')}"
        )


def plot_slice(ax, rewards, objs, picks, labels) -> None:
    """The slice's frontier, with its callouts starred."""
    pareto = rewards[:, objs]
    mop.plot_pareto(
        ax                   = ax,
        pareto               = pareto,
        colors               = COLOR,
        objective            = [labels[o] for o in objs],
        show_dominated       = False,
        nondominated_s       = 30,
        outline_nondominated = 1.0,
        set_lims             = False,
        marker               = "o",
    )
    # A projection's own frontier, sorted along the x objective so the join is monotone.
    front = pareto[get_nondominated(pareto)]
    ax.plot(*front[np.argsort(front[:, 0])].T, color=COLOR, lw=1.5, zorder=0)
    # The callouts, at a size that stays inside the axes at the frontier's ends.
    ax.scatter(*pareto[picks].T, marker="*", s=STAR_SIZE, color=COLOR,
               edgecolors="black", linewidths=1.0, zorder=2)
    ax.margins(0.1)


class Vector(list):
    """A list yaml keeps on one line, so a design reads as a row of numbers."""


yaml.SafeDumper.add_representer(
    Vector,
    lambda dumper, v: dumper.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True),
)


def arm(name: str, design: np.ndarray, tradeoff: np.ndarray) -> dict:
    """One sweep arm: the design its env is built with, and the weights it trains under.

    Rounded to :data:`DECIMALS` so the yaml stays readable. The weights then sum to 1 only
    to that many places, which just rescales a linear scalarization by well under a tenth
    of a percent and leaves the policy it rewards unchanged.
    """
    return {
        "name"                  : name,
        "default_design"        : Vector(round(float(x), DECIMALS) for x in design),
        "default_scalarization" : Vector(round(float(x), DECIMALS) for x in tradeoff),
    }


def sweep_config(robot: str, arms: list[dict]) -> dict:
    """A W&B grid sweep running one arm per callout, as ``cheetah_designs.yaml`` does.

    Each arm pins the design the env is built with and the scalarization its per-objective
    rewards collapse under, which is what makes a PPO run per arm the single-task
    reference for that ``(design, tradeoff)`` pair.
    """
    return {
        "name"   : f"{robot}-ppo-callouts",
        "method" : "grid",
        "metric" : {"name": SWEEP_METRIC, "goal": "maximize"},
        "parameters": {
            "arm"      : {"values": arms},
            "save_dir" : {"value": SWEEP_SAVE_DIR},
        },
    }


def main(args) -> None:
    grid      = load_front(args.robot)
    rewards   = grid.mean_rewards.reshape(-1, grid.n_r)
    designs   = grid.designs.reshape(-1, grid.design_dim)
    tradeoffs = grid.tradeoffs.reshape(-1, grid.n_r)
    labels    = [label.capitalize() for label in codesign.objective_labels(grid.objectives)]
    out_dir   = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms, seen, picked = [], set(), 0
    for objs in combinations(range(grid.n_r), 2):
        picks = slice_callouts(rewards, objs, args.n_callouts)
        picked += len(picks)
        report(labels, list(objs), picks, designs, tradeoffs, rewards)

        name = "-".join(labels[o].lower() for o in objs)
        for rank, i in enumerate(picks, start=1):
            # Neighbouring slices share an endpoint, and that pair needs training once.
            if i in seen:
                continue
            seen.add(i)
            arms.append(arm(f"{name}-{rank}", designs[i], tradeoffs[i]))

        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
        plot_slice(ax, rewards, objs, picks, labels)
        codesign.dress_axis(ax)
        fig.savefig(out_dir / f"callouts_{args.robot}_{name.replace('-', '_')}.{FORMAT}")
        plt.close(fig)

    sweep_path = Path(SWEEP_DIR) / f"{args.robot}_callouts.yaml"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(yaml.safe_dump(
        sweep_config(args.robot, arms), sort_keys=False, default_flow_style=False
    ))
    asked  = args.n_callouts * len(list(combinations(range(grid.n_r), 2)))
    shared = picked - len(arms)
    print(f"\nwrote figures to {out_dir}")
    print(f"wrote {len(arms)} sweep arms to {sweep_path}")
    if shared:
        print(
            f"  {shared} more callouts are endpoints two slices share; an arm's number is "
            f"its rank on its slice, so the numbering skips them"
        )
    if asked > picked:
        print(
            f"  {asked - picked} of the {asked} asked for landed on a point another pick "
            f"already took; a slice cannot be spaced finer than the points it has"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=str, default="cheetah",
                        choices=sorted(icra.FINAL_CONFIGS),
                        help=f"which robot's {ALGO} front to call out pairs on")
    parser.add_argument("--n-callouts", type=int, default=N_CALLOUTS,
                        help="pairs called out per 2D objective slice")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
