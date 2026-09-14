"""Search the design x tradeoff Pareto frontier of a trained MO design network.

Each individual is a ``(design, tradeoff)`` pair, evaluated by rolling the trained policy
``H(d, w)`` out on the model that design builds; the whole population is rolled out in one
batch per generation. ``--algo`` picks NSGA-II or NSGA-III, which differ in how they keep
the frontier spread out -- see ``codesign.optimizers.nsga2.build_algorithm``.

Writes two datasets in the repository's grid format, one pair per cell:
``<algo>_archive.npz`` (every pair evaluated) and ``<algo>_front.npz`` (the non-dominated
ones). ``plot_nsga2_front.py`` reads them back.

The search itself lives in ``codesign.optimizers.nsga2``.
"""

import argparse
from pathlib import Path

import minimal_mjx as mm
import moplayground as mop

import codesign
from codesign.optimizers.nsga2 import ALGORITHMS, run_nsga

ENTITY          = "vmadabushi3-georgia-institute-of-technology"
PROJECT         = "codesign"
ARTIFACT_PREFIX = "hypernetworks"

# The trained run each network defaults to searching against.
RUN_IDS = {"mdh": "zpzu9ms5", "mlp": "dkfq4g7t"}

DOWNLOAD_DIR = "results/wandb-downloads"
SAVE_PATH    = "scripts/icra/outputs"

POP_SIZE = 32    # pairs rolled out together, i.e. the batch width
BUDGET   = 1024  # total pairs evaluated, so BUDGET / POP_SIZE generations
STEPS    = 500   # rollout length (env steps)


def load_config(args) -> dict:
    """Resolve the run config, downloading the W&B run's checkpoint first unless given one."""
    if args.config is not None:
        return mm.create_config_dict(mop.utils.read_config(args.config))

    run_id = args.run_id or RUN_IDS[args.network]
    mm.download_model(
        run_id     = run_id,
        save_dir   = args.download_dir,
        model_name = run_id,
        entity     = args.entity,
        project    = args.project,
        prefix     = ARTIFACT_PREFIX,
    )
    # download_model writes the run's config next to the checkpoint it fetched.
    config = mm.read_config(Path(args.download_dir) / run_id / "config.yaml")
    return mm.create_config_dict(config)


def main(args) -> None:
    config = load_config(args)
    env, _ = codesign.load_env(config)

    front, archive, _ = run_nsga(
        env, config,
        n_steps         = args.steps,
        budget          = args.budget,
        pop_size        = args.pop_size,
        algorithm       = args.algo,
        per_cell        = args.per_cell,
        checkpoint_path = args.checkpoint,
        seed            = args.seed,
        deterministic   = not args.stochastic_policy,
        workers         = args.workers,
        verbose         = not args.quiet,
    )

    save_dir = Path(args.save_path) / (args.run_id or RUN_IDS.get(args.network, config.name))
    save_dir.mkdir(parents=True, exist_ok=True)
    archive.save(save_dir / f"{args.algo}_archive.npz")
    front.save(save_dir / f"{args.algo}_front.npz")
    print(
        f"{args.algo} on {config.algorithm}: {archive.n_designs} pairs evaluated, "
        f"{front.n_designs} non-dominated -> {save_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--network", type=str, default="mdh", choices=sorted(RUN_IDS),
                        help=f"which trained network to search against, by its default "
                             f"W&B run id ({', '.join(f'{k}={v}' for k, v in RUN_IDS.items())})")
    source.add_argument("--run_id", type=str, default=None,
                        help="W&B run id to download and search against, overriding --network")
    source.add_argument("--config", type=str, default=None,
                        help="path to a local config.yaml of an already-downloaded run; "
                             "skips the download")
    parser.add_argument("--save_path", type=str, default=SAVE_PATH,
                        help="directory the datasets are written into")
    parser.add_argument("--download_dir", type=str, default=DOWNLOAD_DIR,
                        help="directory the W&B run downloads into")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit checkpoint dir; defaults to the latest of the run")
    parser.add_argument("--algo", type=str, default="nsga2", choices=ALGORITHMS,
                        help="nsga2 keeps the frontier spread by crowding distance; "
                             "nsga3 by reference directions, which holds up better at "
                             "three or more objectives")
    parser.add_argument("--pop_size", type=int, default=POP_SIZE,
                        help=f"pairs rolled out together each generation (default "
                             f"{POP_SIZE}); a power of two keeps the Sobol initial "
                             "population balanced")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help=f"total pairs evaluated over the whole search (default "
                             f"{BUDGET}), i.e. budget / pop_size generations")
    parser.add_argument("--per_cell", type=int, default=1,
                        help="rollout repetitions per pair, averaged into its objective "
                             "vector; only informative with --stochastic_policy")
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="rollout length (env steps)")
    parser.add_argument("--workers", type=int, default=1,
                        help="threads compiling the per-design mjx models")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic_policy", action="store_true",
                        help="sample the policy each step instead of taking its mode")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress pymoo's per-generation table")
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument("--project", type=str, default=PROJECT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
