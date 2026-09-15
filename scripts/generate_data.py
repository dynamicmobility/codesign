"""Generate rollout datasets from a trained design hypernetwork, MO design hypernetwork,
or morlax run.

Each run's policy is swept over the axes it was trained to condition on -- designs for
``design_hypernetwork``, designs x tradeoffs for ``mo_design_hypernetwork``, tradeoffs
alone for ``morlax`` -- and the rolled-out grid is saved as one npz.
"""

import argparse
from pathlib import Path

import minimal_mjx as mm
import moplayground as mop
import codesign
import os

ENTITY            = os.environ["WANDB_ENTITY"]
PROJECT           = "codesign"
ARTIFACT_PREFIX   = "hypernetworks"

DOWNLOAD_DIR      = "results/wandb-downloads"
SAVE_PATH         = None #"scripts/outputs/datasets"

N_DESIGNS   = 16   # designs the sweep covers the design box with
N_TRADEOFFS = 16   # sampled tradeoffs; == num_objectives gives the simplex corners
PER_CELL    = 1    # rollout repetitions per (design, tradeoff) cell
STEPS       = 500  # rollout length (env steps)
RECORD      = ("reward", "done", "qpos")
TRADEOFFS   = "uniform"

# The rollout each supported algorithm sweeps its grid with.
ROLLOUTS = {
    "design_hypernetwork"    : codesign.rollout_design_hypernetwork_grid,
    "mo_design_hypernetwork" : codesign.rollout_mo_design_hypernetwork,
    "morlax"                 : codesign.rollout_morlax,
}

# Sweep axes an algorithm does not have, so their options must be left off the command
# line: the single-objective hypernetwork trains under one fixed scalarization, and morlax
# trains on one fixed design.
UNUSED_OPTIONS = {
    "design_hypernetwork"    : ("n_tradeoffs", "tradeoffs"),
    "mo_design_hypernetwork" : (),
    "morlax"                 : ("n_designs",),
}

# Defaults for the options an algorithm does use; the rest stay ``None``.
DEFAULTS = {
    "n_designs"   : N_DESIGNS,
    "n_tradeoffs" : N_TRADEOFFS,
    "tradeoffs"   : TRADEOFFS,
}


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


def check_options(config, args) -> None:
    """Reject a run this script cannot sweep, and options its algorithm has no axis for."""
    algorithm = config.algorithm
    if algorithm not in ROLLOUTS:
        raise ValueError(
            f"'{algorithm}' runs cannot be swept; expected one of {sorted(ROLLOUTS)}."
        )
    for option in UNUSED_OPTIONS[algorithm]:
        if getattr(args, option) is not None:
            raise ValueError(
                f"--{option} does not apply to '{algorithm}', which has no such sweep "
                "axis; drop the option."
            )


def fill_defaults(config, args):
    """Default the options the algorithm does use. They parse as ``None`` so that passing
    one it has no axis for is an error rather than a silently ignored value."""
    for option, default in DEFAULTS.items():
        if option not in UNUSED_OPTIONS[config.algorithm] and getattr(args, option) is None:
            setattr(args, option, default)
    return args


def build_grid(config, env, args) -> codesign.Grid:
    """The grid the run's algorithm is swept over, in physical design units."""
    # The sampled tradeoffs of a multi-objective run; a single-objective one leaves the
    # layout unset (see UNUSED_OPTIONS) and sweeps its own scalarization instead.
    tradeoffs = None if args.tradeoffs is None else codesign.tradeoff_layout(
        args.tradeoffs, len(env.objectives), args.seed, n_tradeoffs=args.n_tradeoffs
    )
    match config.algorithm:
        case "design_hypernetwork":
            return codesign.Grid.from_design_sample(
                env,
                args.seed,
                n_designs  = args.n_designs,
                per_cell   = args.per_cell,
                tradeoffs  = [config.env_config.reward.optimization.default_scalarization],
                objectives = env.objectives,
            )
        case "mo_design_hypernetwork":
            return codesign.Grid.from_design_sample(
                env,
                args.seed,
                n_designs  = args.n_designs,
                per_cell   = args.per_cell,
                tradeoffs  = tradeoffs,
                objectives = env.objectives,
            )
        case "morlax":
            # morlax trains on one design, so the design axis is that design alone.
            return codesign.Grid.crossed(
                [config.env_config.codesign.default_design],
                tradeoffs,
                per_cell   = args.per_cell,
                objectives = env.objectives,
            )


def dataset_name(args) -> str:
    """``sweep`` for the default tradeoff layout, which the plotting scripts load."""
    if args.tradeoffs in (None, TRADEOFFS):
        return "sweep"
    return f"sweep_{args.tradeoffs}"


def main(args) -> None:
    config = load_config(args)
    check_options(config, args)
    args = fill_defaults(config, args)

    env, _ = codesign.load_env(config)
    dataset = ROLLOUTS[config.algorithm](
        env             = env,
        config          = config,
        grid            = build_grid(config, env, args),
        n_steps         = args.steps,
        checkpoint_path = args.checkpoint,
        seed            = args.seed,
        deterministic   = not args.stochastic_policy,
        record          = args.record,
    )
    
    if args.save_path is None:
        if args.config:
            save_path = Path(args.config).parent
        else:
            save_path = Path(DOWNLOAD_DIR) / args.run_id
    else:
        save_path = Path(args.save_path) / (args.run_id or config.name)
    save_path.mkdir(parents=True, exist_ok=True)
    path = save_path / f"{dataset_name(args)}.npz"
    dataset.save(path)
    print(f"rewards {dataset.rewards.shape}, keys {dataset.keys} -> {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run_id", type=str, default=None,
                        help="W&B run id to download the checkpoint + config from")
    source.add_argument("--config", type=str, default=None,
                        help="path to a local config.yaml of an already-downloaded run")
    parser.add_argument("--save_path", type=str, default=SAVE_PATH,
                        help="directory the dataset is written into")
    parser.add_argument("--download_dir", type=str, default=DOWNLOAD_DIR,
                        help="directory --run_id downloads into")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit checkpoint dir; defaults to the latest of the run")
    parser.add_argument("--n_designs", type=int, default=None,
                        help=f"designs the sweep covers the design box with (default "
                             f"{N_DESIGNS}); morlax trains on one fixed design, so it "
                             "takes no such option")
    parser.add_argument("--n_tradeoffs", type=int, default=None,
                        help=f"tradeoffs of the sweep (default {N_TRADEOFFS}); per "
                             "objective pair under --tradeoffs 2d, ignored under "
                             "--tradeoffs corners. Single-objective runs take no tradeoff")
    parser.add_argument("--tradeoffs", type=str, default=None,
                        choices=sorted(codesign.TRADEOFF_LAYOUTS),
                        help=f"tradeoff layout of the sweep (default {TRADEOFFS}): "
                             "'uniform' samples the simplex, 'corners' takes its one-hot "
                             "corners, '2d' sweeps each objective pair's edge. Non-default "
                             "layouts are saved as sweep_<layout>.npz")
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
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
