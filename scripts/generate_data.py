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

# Tradeoff layout of the non-predictor grid -> its DesignTradeoffSampleGrid constructor
# and the keyword --n_tradeoffs feeds it (None where the layout fixes its own count).
TRADEOFF_LAYOUTS = {
    "uniform" : (codesign.DesignTradeoffSampleGrid.from_uniform_sample, "n_tradeoffs"),
    "corners" : (codesign.DesignTradeoffSampleGrid.from_simplex_corners, None),
    "2d"      : (codesign.DesignTradeoffSampleGrid.from_2d_tradeoffs, "n_tradeoffs_per_pair"),
}
TRADEOFFS = "uniform"


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


def build_sweep_grid(env, args) -> codesign.DesignTradeoffSampleGrid:
    """The non-predictor grid: a Sobol design sweep crossed with ``--tradeoffs``.

    All three layouts sweep the design box the same way and differ only in which
    tradeoffs they cross it with: ``uniform`` samples the whole simplex, ``corners``
    takes just its one-hot corners (one per objective), and ``2d`` sweeps the edge of
    every objective pair, ``--n_tradeoffs`` weights per pair.
    """
    build, count_arg = TRADEOFF_LAYOUTS[args.tradeoffs]
    return build(
        env,
        seed      = args.seed,
        n_designs = args.n_designs,
        per_cell  = args.per_cell,
        **({count_arg: args.n_tradeoffs} if count_arg else {}),
    )


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

    def build_grid(design_predictor: bool):
        """The design-box sweep, or the predictor's own designs for each tradeoff."""
        if not design_predictor:
            return build_sweep_grid(env, args)
        # params = (normalizer, hypernet, design_predictor)
        _, design_predictor_inference_fn, params = (
            codesign.load_mo_design_predictor_hypernetwork(config, path=args.checkpoint)
        )
        return codesign.DesignPredictorSampleGrid.from_predictor(
            env,
            design_predictor_inference_fn,
            params[2],
            seed        = args.seed,
            n_tradeoffs = args.n_tradeoffs,
            group_size  = args.n_designs,
            per_cell    = args.per_cell,
        )

    def rollout(design_predictor: bool) -> codesign.DesignTradeoffDataset:
        return codesign.rollout_mo_design_hypernetwork(
            env              = env,
            config           = config,
            grid             = build_grid(design_predictor),
            n_steps          = args.steps,
            checkpoint_path  = args.checkpoint,
            seed             = args.seed,
            deterministic    = not args.stochastic_policy,
            record           = args.record,
        )

    datasets = {}
    if not args.predictor_only:
        # the default layout keeps the plain `sweep` name the plotting scripts load
        sweep_name = "sweep" + ("" if args.tradeoffs == TRADEOFFS else f"_{args.tradeoffs}")
        datasets[sweep_name] = rollout(design_predictor=False)
    if is_predictor_run:
        datasets["predictor"] = rollout(design_predictor=True)

    if args.run_id:
        save_path = save_path / args.run_id
    else:
        save_path = save_path / config.name
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, dataset in datasets.items():
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
                        help="tradeoffs of the design sweep; per objective pair under "
                             "--tradeoffs 2d, ignored under --tradeoffs corners")
    parser.add_argument("--tradeoffs", type=str, default=TRADEOFFS,
                        choices=sorted(TRADEOFF_LAYOUTS),
                        help="tradeoff layout of the design sweep: 'uniform' samples the "
                             "simplex, 'corners' takes its one-hot corners, '2d' sweeps "
                             "each objective pair's edge. Non-default layouts are saved "
                             "as sweep_<layout>.npz")
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
