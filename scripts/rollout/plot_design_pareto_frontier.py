"""
Loads the reward data from the .npz and plots it
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
from codesign.eval import rollout_mo_designs

CONFIG_PATH = "config/mo_design_hypernetwork_cheetah.yaml"
DATA_PATH = "scripts/outputs/mo_design_hypernetwork_rewards.npz"

N_DESIGNS = 8           # number of designs in the uniform sweep
N_TRADEOFFS = 8          # number of tradeoffs in the uniform sweep
T = 2          # rollout length (env steps)
OUT_DIR = Path("scripts/outputs")


def main(config_path: str, checkpoint_path: str | None, n_designs: int, n_tradeoffs: int, steps: int) -> None:
    config     = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env        = MAICheetah(env_params=env_params, backend="jnp")

    data_path = Path(DATA_PATH)
    with np.load(data_path, allow_pickle=True) as data:
        reward_data = {k: data[k] for k in data.files}

    print(f"Loaded {data_path} with keys: {list(reward_data.keys())}")
    print(reward_data['reward']) # This is N_DESIGNS x N_TRADEOFFS x num_objectives array of rewards

    num_objectives = reward_data["reward"].shape[-1]
    print(reward_data["designs"])


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
