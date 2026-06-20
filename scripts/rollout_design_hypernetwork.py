"""Evaluate a trained ``design_hypernetwork`` across a uniform sweep of designs
across the back leg. Creates two figures:

  1. reward vs. time, one translucent line per design (coloured by design value);
  2. a bar chart of cumulative reward per design.

The parallel rollout itself lives in ``codesign.eval.rollout_design_hypernetwork``; 
this script only builds the env, calls it, and plots the result.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import minimal_mjx as mm
import moplayground as mop
from codesign.envs.MAICheetah import MAICheetah
from codesign.eval import rollout_design_hypernetwork

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"

N = 64           # number of designs in the uniform sweep
T = 500          # rollout length (env steps)
OUT_DIR = Path("scripts/outputs")


def main(config_path: str, checkpoint_path: str | None, n: int, steps: int) -> None:
    config     = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env        = MAICheetah(env_params=env_params, backend="jnp")

    # Parallel rollout of the trained hypernetwork across the design sweep.
    designs, rewards = rollout_design_hypernetwork(
        env             = env,
        config          = config,
        num_envs        = n,
        n_steps         = steps,
        checkpoint_path = checkpoint_path
    )
    cumulative = rewards.sum(axis=0)  # (N,)
    time       = np.arange(steps)

    design      = config["learning_params"]["design_params"]
    design_low  = float(design["design_low"])
    design_high = float(design["design_high"])

    # Save logged rewards.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "design_hypernetwork_rewards.npz",
        time=time, rewards=rewards, cumulative=cumulative, designs=designs,
    )

    # Reward vs. time, one line per design (coloured by design value).
    norm = Normalize(vmin=design_low, vmax=design_high)
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n):
        ax.plot(time, rewards[:, i], color=cmap(norm(designs[i, 0])), alpha=0.4, lw=0.8)
    ax.set_xlabel("env step")
    ax.set_ylabel("scalar reward")
    ax.set_title(f"design_hypernetwork reward over time ({n} designs, {steps} steps)")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="design d (back-leg length scale)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "design_hypernetwork_reward_over_time.png", dpi=150)
    plt.close(fig)

    # Cumulative reward per design (bar chart).
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [cmap(norm(d)) for d in designs[:, 0]]
    width = (design_high - design_low) / n * 0.9
    ax.bar(designs[:, 0], cumulative, width=width, color=bar_colors)
    ax.set_xlabel("design d (back-leg length scale)")
    ax.set_ylabel(f"cumulative reward over {steps} steps")
    ax.set_title(f"design_hypernetwork cumulative reward by design ({n} designs)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "design_hypernetwork_cumulative_reward.png", dpi=150)
    plt.close(fig)

    print(
        f"rewards {rewards.shape} | cumulative mean {cumulative.mean():.2f} "
        f"std {cumulative.std():.2f} -> {OUT_DIR}/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="explicit checkpoint dir; defaults to latest under save_dir/name",
    )
    parser.add_argument("--n", type=int, default=N, help="number of designs in the sweep")
    parser.add_argument("--steps", type=int, default=T, help="rollout length (env steps)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.n, args.steps)
