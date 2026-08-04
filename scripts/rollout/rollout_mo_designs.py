"""Evaluate a trained ``mo_design_hypernetwork`` across a uniform sweep of designs
and tradeoffs.
Logs the results in a ``.npz`` file for later plotting.
The parallel rollout itself lives in ``codesign.eval.rollout_mo_design_hypernetwork``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import minimal_mjx as mm
import moplayground as mop
import codesign

CONFIG_PATH = "config/mo_design_hypernetwork_cheetah.yaml"

N_DESIGNS   = 16          # number of designs in the uniform sweep
N_TRADEOFFS = 16          # number of tradeoffs in the uniform sweep
T = 5                     # rollout length (env steps)
OUT_DIR = Path("scripts/outputs")


def plot_mo_designs_rollout(
    grid: codesign.DesignTradeoffRolloutGrid,
    save_dir: Path = None,
    show_dominated_designs=True,
):
    """Produce the per-design frontier plot and the design-objective frontier plot."""
    rewards = grid.mean_rewards
    num_obj = rewards.shape[-1]
    projection = "3d" if num_obj == 3 else None
    # projection = None

    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(111, projection=projection)
    codesign.plot_design_paretos(
        ax1, rewards, grid.objectives, grid.designs, show_dominated=show_dominated_designs
    )
    ax1.set_title("Per-design Pareto frontiers")
    ax1.legend(fontsize=8)
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111, projection=projection)
    codesign.plot_design_objective_pareto(ax2, rewards, grid.objectives, grid.designs)
    ax2.set_title("Design-objective Pareto frontier")
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        fig1.savefig(save_dir / "rollout_design_paretos.png", dpi=150)
        fig2.savefig(save_dir / "rollout_design_objective_pareto.png", dpi=150)
    return (fig1, ax1), (fig2, ax2)


def main(
    config_path: str,
    checkpoint_path: str | None,
    n_designs: int,
    n_tradeoffs: int,
    steps: int,
    npz_path: str | None = None,
) -> None:
    if npz_path is not None:
        grid = codesign.DesignTradeoffRolloutGrid.load(npz_path)
        print(f"Loaded rollout from {npz_path} (rewards shape {grid.rewards.shape})")
    else:
        config     = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
        env_params = mm.utils.config.create_config_dict(config["env_config"])
        env        = codesign.cheetah(env_params=env_params, backend="jnp")

        grid = codesign.rollout_mo_design_hypernetwork(
            env             = env,
            config          = config,
            n_designs       = n_designs,
            n_tradeoffs     = n_tradeoffs,
            per_cell        = 1,
            n_steps         = steps,
            checkpoint_path = checkpoint_path,
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        grid.save(OUT_DIR / "mo_design_hypernetwork_rewards.npz")

    plot_mo_designs_rollout(grid, save_dir=OUT_DIR)
    plt.close("all")


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
    parser.add_argument(
        "--npz", type=str, default=None,
        help="path to a previously saved rollout .npz; if given, skips rolling out again "
             "and plots directly from the logged grid",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.n_designs, args.n_tradeoffs, args.steps, args.npz)
