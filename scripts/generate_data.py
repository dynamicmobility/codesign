"""Generate design x tradeoff rollout datasets from a trained MO design hypernetwork.
"""

import argparse
from pathlib import Path

import minimal_mjx as mm
import moplayground as mop

import codesign

ENTITY            = "vmadabushi3-georgia-institute-of-technology"
PROJECT           = "codesign"
ARTIFACT_PREFIX   = "hypernetworks"

DOWNLOAD_DIR          = "results/wandb-downloads"
SAVE_PATH             = "scripts/outputs/datasets"
PREDICTOR_ALGORITHM   = "mo_design_predictor_hypernetwork"

N_DESIGNS   = 16   # designs per tradeoff (the whole sweep, for the non-predictor grid)
N_TRADEOFFS = 16   # sampled tradeoffs; == num_objectives gives the simplex corners
PER_CELL    = 1    # rollout repetitions per (design, tradeoff) cell
STEPS       = 500  # rollout length (env steps)
RECORD      = ("reward", "done", "qpos")


def load_config(args) -> dict:
    """Resolve the run config, downloading the W&B run's checkpoint first if asked."""
    if args.run_id is not None:
        mm.download_model(
            run_id     = args.run_id,
            save_dir   = args.download_dir,
            model_name = args.run_id,
            entity     = ENTITY,
            project    = PROJECT,
            prefix     = ARTIFACT_PREFIX,
        )
        # download_model writes the run's config next to the checkpoint it fetched.
        config = mm.read_config(Path(args.download_dir) / args.run_id / "config.yaml")
    else:
        config = mop.utils.read_config(args.config)
    return mm.create_config_dict(config)


def main(args) -> None:
    config = load_config(args)
    env, _ = codesign.envs.create.load_env(config)
    is_predictor_run = config["algorithm"] == PREDICTOR_ALGORITHM
    if args.predictor_only and not is_predictor_run:
        raise ValueError(
            f"--predictor_only needs a '{PREDICTOR_ALGORITHM}' run; this one is "
            f"'{config['algorithm']}', which has no design predictor to sample from."
        )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    def rollout(design_predictor: bool) -> codesign.DesignTradeoffDataset:
        return codesign.rollout_mo_design_hypernetwork(
            env              = env,
            config           = config,
            n_designs        = args.n_designs,
            n_tradeoffs      = args.n_tradeoffs,
            per_cell         = args.per_cell,
            n_steps          = args.steps,
            checkpoint_path  = args.checkpoint,
            seed             = args.seed,
            deterministic    = not args.stochastic_policy,
            design_predictor = design_predictor,
            record           = args.record,
        )

    datasets = {}
    if not args.predictor_only:
        datasets["sweep"] = rollout(design_predictor=False)
    if is_predictor_run:
        datasets["predictor"] = rollout(design_predictor=True)

    for name, dataset in datasets.items():
        if args.run_id:
            save_path = save_path / args.run_id
        else:
            save_path = save_path / config.name
        dataset.save(save_path / f"{name}.npz")
        print(f"{name}: rewards {dataset.rewards.shape}, keys {dataset.keys} -> "
              f"{save_path / f'{name}.npz'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run_id", type=str, default=None,
                        help="W&B run id to download the checkpoint + config from")
    source.add_argument("--config", type=str, default=None,
                        help="path to a local config.yaml of an already-downloaded run")
    parser.add_argument("--save_path", type=str, default=SAVE_PATH,
                        help="directory the datasets are written into")
    parser.add_argument("--download_dir", type=str, default=DOWNLOAD_DIR,
                        help="directory --run_id downloads into")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit checkpoint dir; defaults to the latest of the run")
    parser.add_argument("--n_designs", type=int, default=N_DESIGNS,
                        help="designs per tradeoff; 1 takes the predictor's mode")
    parser.add_argument("--n_tradeoffs", type=int, default=N_TRADEOFFS,
                        help="sampled tradeoffs; equal to the objective count gives the "
                             "one-hot corners of the simplex")
    parser.add_argument("--per_cell", type=int, default=PER_CELL,
                        help="repetitions per cell; only informative with a stochastic "
                             "policy or env reset")
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="rollout length (env steps)")
    parser.add_argument("--record", type=str, nargs="+", default=list(RECORD),
                        choices=sorted(codesign.TRAJECTORY_FIELDS),
                        help="per-step fields to keep in dataset.data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic_policy", action="store_true",
                        help="sample the policy each step instead of taking its mode")
    parser.add_argument("--predictor_only", action="store_true",
                        help="skip the design-box sweep and only roll out the predictor's designs")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
