"""Evaluate a trained ``mo_design_hypernetwork`` across a uniform sweep of designs
and tradeoffs. 
Logs the results in a ``.npz`` file for later plotting.
The parallel rollout itself lives in ``codesign.eval.rollout_parallel``; 
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import minimal_mjx as mm
import moplayground as mop
from codesign.envs.CodesignCheetah import CodesignCheetah
from codesign.eval import rollout_mo_designs
import pdb

CONFIG_PATH = "config/mo_design_hypernetwork_cheetah.yaml"

N_DESIGNS = 8           # number of designs in the uniform sweep
N_TRADEOFFS = 8          # number of tradeoffs in the uniform sweep
T = 2          # rollout length (env steps)
OUT_DIR = Path("scripts/outputs")


def main(config_path: str, checkpoint_path: str | None, n_designs: int, n_tradeoffs: int, steps: int) -> None:
    config     = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env        = CodesignCheetah(env_params=env_params, backend="jnp")

    # Parallel rollout of the trained hypernetwork across the design sweep.
    designs, final_state, final_reward, states = rollout_mo_designs(
        env             = env,
        config          = config,
        n_designs       = n_designs,
        n_tradeoffs     = n_tradeoffs,
        per_cell        = 1,
        n_steps         = steps,
        checkpoint_path = checkpoint_path
    )
    print(final_reward)
    time       = np.arange(steps)

    design      = config["learning_params"]["design_params"]
    design_low  = float(design["design_low"])
    design_high = float(design["design_high"])

    # Save logged rewards.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "mo_design_hypernetwork_rewards.npz",
        time=time, reward=final_reward, designs=designs,
    )

    # # Reward vs. time, one line per design (coloured by design value).
    # norm = Normalize(vmin=design_low, vmax=design_high)
    # cmap = cm.viridis
    # fig, ax = plt.subplots(figsize=(9, 5))
    # for i in range(n_designs):
    #     ax.plot(time, rewards[:, i], color=cmap(norm(designs[i, 0])), alpha=0.4, lw=0.8)
    # ax.set_xlabel("env step")
    # ax.set_ylabel("scalar reward")
    # ax.set_title(f"design_hypernetwork reward over time ({n_designs} designs, {steps} steps)")
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax, label="design d (back-leg length scale)")
    # fig.tight_layout()
    # fig.savefig(OUT_DIR / "design_hypernetwork_reward_over_time.png", dpi=150)
    # plt.close(fig)

    # # Cumulative reward per design (bar chart).
    # fig, ax = plt.subplots(figsize=(10, 5))
    # bar_colors = [cmap(norm(d)) for d in designs[:, 0]]
    # width = (design_high - design_low) / n_designs * 0.9
    # ax.bar(designs[:, 0], cumulative, width=width, color=bar_colors)
    # ax.set_xlabel("design d (back-leg length scale)")
    # ax.set_ylabel(f"cumulative reward over {steps} steps")
    # ax.set_title(f"design_hypernetwork cumulative reward by design ({n_designs} designs)")
    # fig.tight_layout()
    # fig.savefig(OUT_DIR / "design_hypernetwork_cumulative_reward.png", dpi=150)
    # plt.close(fig)

    # # Show reward components

    # print(
    #     f"rewards {rewards.shape} | cumulative mean {cumulative.mean():.2f} "
    #     f"std {cumulative.std():.2f} -> {OUT_DIR}/"
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="explicit checkpoint dir; defaults to latest under save_dir/name",
    )
    parser.add_argument("--n_designs", type=int, default=N_DESIGNS, help="number of designs in the sweep")
    parser.add_argument("--n_tradeoffs", type=int, default=N_TRADEOFFS, help="number of tradeoffs in the sweep")
    parser.add_argument("--steps", type=int, default=T, help="rollout length (env steps)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.n_designs, args.n_tradeoffs, args.steps)
