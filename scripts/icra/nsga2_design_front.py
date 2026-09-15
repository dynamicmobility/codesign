"""Search the design x tradeoff Pareto frontier of every trained network a robot has.


Run from the repository root, where the configured paths resolve:
``python -m scripts.icra.nsga2_design_front --robot cheetah`` --algorithm MDH.
"""

import argparse
from pathlib import Path

import minimal_mjx as mm

import codesign
from codesign.optimizers.nsga2 import run_nsga

from scripts import icra

ENTITY          = "vmadabushi3-georgia-institute-of-technology"
PROJECT         = "codesign"
ARTIFACT_PREFIX = "hypernetworks"
DOWNLOAD_DIR    = "results/wandb-downloads"

SEARCH_ALGO   = "nsga3"  # names the datasets written, as well as picking the search
STEPS         = 500      # rollout length (env steps)
PER_CELL      = 1        # rollouts per pair; only informative with a stochastic policy
SEED          = 0        # PRNG seed for the rollouts and for pymoo's operators
DETERMINISTIC = True     # take the policy's mode each step rather than sampling it
VERBOSE       = True     # print pymoo's per-generation table


def download_run(run: icra.Run) -> tuple[dict, Path]:
    """Fetch the run from W&B: its config, and its first checkpoint at or past
    ``run.checkpoint`` training steps.

    Returns the config dict and the directory that checkpoint downloaded into.
    """
    checkpoint = mm.download_model(
        run_id     = run.run_id,
        save_dir   = DOWNLOAD_DIR,
        model_name = run.run_id,
        entity     = ENTITY,
        project    = PROJECT,
        prefix     = ARTIFACT_PREFIX,
        iterations = run.checkpoint,
    )
    # download_model writes the run's config next to the checkpoint it fetched.
    config = mm.read_config(Path(DOWNLOAD_DIR) / run.run_id / "config.yaml")
    return mm.create_config_dict(config), Path(checkpoint)


def search(algo: str, run: icra.Run, settings: dict) -> None:
    """Search one trained network's frontier and write its two datasets."""
    config, checkpoint = download_run(run)
    env, _ = codesign.load_env(config)

    front, archive, _ = run_nsga(
        env, config,
        n_steps         = STEPS,
        budget          = settings["budget"],
        pop_size        = settings["pop_size"],
        algorithm       = SEARCH_ALGO,
        per_cell        = PER_CELL,
        checkpoint_path = checkpoint,
        seed            = SEED,
        deterministic   = DETERMINISTIC,
        workers         = settings["workers"],
        verbose         = VERBOSE,
    )

    save_dir = Path(run.dataset_path) / run.run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    archive.save(save_dir / f"{SEARCH_ALGO}_archive.npz")
    front.save(save_dir / f"{SEARCH_ALGO}_front.npz")
    print(
        f"{algo} ({run.run_id} @ step {checkpoint.name}): {archive.n_designs} pairs "
        f"evaluated, {front.n_designs} non-dominated -> {save_dir}"
    )


def main(args) -> None:
    runs = icra.FINAL_CONFIGS[args.robot]
    if args.algorithm is not None:
        if args.algorithm not in runs:
            raise SystemExit(
                f"{args.robot} configures no '{args.algorithm}'; it has {sorted(runs)}"
            )
        runs = {args.algorithm: runs[args.algorithm]}

    settings = icra.DATA_GEN[args.robot]
    for algo, run in runs.items():
        # Only the algorithms already trained carry a Run; the rest name no checkpoint.
        if not isinstance(run, icra.Run):
            print(f"warning: {algo} has no run configured; skipping")
            continue
        search(algo, run, settings)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=str, default="cheetah",
                        choices=sorted(icra.FINAL_CONFIGS),
                        help="which robot's configured runs to search")
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=sorted({a for r in icra.FINAL_CONFIGS.values() for a in r}),
                        help="search only this algorithm's run; the default searches "
                             "every algorithm the robot configures a run for")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
