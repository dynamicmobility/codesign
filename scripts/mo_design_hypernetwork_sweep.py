"""Grid hyperparameter search for the MO design hypernetwork on CodesignCheetah.

Trial ``i`` of the grid trains under ``save_dir/sweep_name/sweep_i`` and is scored by
the Pareto statistics ``plot_cum_hv_progress`` accumulates over its evals. W&B runs are
named ``sweep_i`` and grouped under ``sweep_name``. Results are written to
``save_dir/sweep_name/sweep.csv`` after every trial.
"""

import argparse
import copy
import functools
import itertools
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import minimal_mjx as mm
import moplayground as mop

from codesign.envs.EnvLoader import load_env
from codesign.hyperdesigners import setup_mo_design_hypernetwork
from codesign.utils.plotting import MODesignTrainingPlottingInfo, plot_cum_hv_progress

CONFIG_PATH = "config/mo_design_hypernetwork_cheetah.yaml"
WANDB_ENTITY = "vmadabushi3-georgia-institute-of-technology"

# Dotted config paths -> values swept over. The trial set is their full product.
GRID = {
    "learning_params.ppo_params.learning_rate":    [1e-4, 5e-4, 1e-3],
    # "learning_params.ppo_params.entropy_cost":     [1e-3, 1e-2],
    # "learning_params.network_params.num_features": [16, 32, 64],
    "learning_params.ppo_params.batch_size": [32, 64, 128],
    "learning_params.ppo_params.num_minibatches": [32, 16, 8],
    # "learning_params.network_params.num_features": [16, 32, 64],
    # "learning_params.tradeoff_params.sampling":    ["sparse-heavytail"],
}


def set_path(config: dict, path: str, value):
    *parents, leaf = path.split(".")
    for key in parents:
        config = config[key]
    config[leaf] = value


def slug(value) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def trial_columns(overrides: dict) -> dict:
    return {path.split(".")[-1]: slug(value) for path, value in overrides.items()}


def rejection_reason(config: dict) -> str | None:
    """The trainer's shape constraint violated by ``config``, if any."""
    ppo = config["learning_params"]["ppo_params"]
    cells = (
        config["learning_params"]["design_params"]["num_designs"]
        * config["learning_params"]["tradeoff_params"]["num_tradeoffs"]
    )
    if ppo["num_envs"] % cells:
        return f"num_envs={ppo['num_envs']} not divisible by {cells} grid cells"
    if ppo["num_eval_envs"] % cells:
        return f"num_eval_envs={ppo['num_eval_envs']} not divisible by {cells} grid cells"
    if (ppo["batch_size"] * ppo["num_minibatches"]) % ppo["num_envs"]:
        return "batch_size * num_minibatches not divisible by num_envs"
    return None


def build_trials(
    base: dict, sweep_name: str, num_timesteps: int | None
) -> list[tuple[dict, dict]]:
    """``(config, overrides)`` per point of the grid, named ``sweep_name/sweep_i``."""
    trials = []
    for i, combo in enumerate(itertools.product(*GRID.values())):
        overrides = dict(zip(GRID, combo))
        config = copy.deepcopy(base)
        for path, value in overrides.items():
            set_path(config, path, value)
        if num_timesteps is not None:
            config["learning_params"]["ppo_params"]["num_timesteps"] = num_timesteps
        config["name"] = f"{sweep_name}/sweep_{i}"
        trials.append((config, overrides))
    return trials


def run_trial(
    config: dict, overrides: dict, sweep_name: str, wandb_project: str | None
) -> dict:
    env, _ = load_env(config)
    eval_env, _ = load_env(config)

    run = None
    if wandb_project:
        run = mm.utils.logging.initialize_wandb(
            name    = Path(config["name"]).name,
            entity  = WANDB_ENTITY,
            project = wandb_project,
            config  = config,
            group   = sweep_name,
            reinit  = "create_new",
        )

    training_data = MODesignTrainingPlottingInfo(
        start_time = time.time(),
        labels     = env.objectives,
    )
    start = time.time()
    mm.learning.training.train(
        config,
        env,
        eval_env,
        run           = run,
        handle_params = setup_mo_design_hypernetwork,
        progress_fn   = functools.partial(
            plot_cum_hv_progress, training_data=training_data
        ),
    )
    if run:
        run.finish()

    hvs = np.asarray(training_data.aux["hv"])
    sps = np.asarray(training_data.aux["sp"])
    return {
        **trial_columns(overrides),
        "name"          : config["name"],
        "cum_hv"        : float(hvs.sum()),
        "final_hv"      : float(hvs[-1]),
        "mean_spacing"  : float(sps.mean()),
        "final_spacing" : float(sps[-1]),
        "walltime"      : time.time() - start,
        "error"         : "",
    }


def main(
    config_path: str,
    sweep_name: str | None = None,
    num_timesteps: int | None = None,
    dry_run: bool = False,
    wandb_project: str | None = None,
    rerun_existing: bool = False,
):
    base = mop.utils.read_config(config_path)
    sweep_name = sweep_name or f"{base['name']}_sweep"
    trials = build_trials(base, sweep_name, num_timesteps)
    sweep_dir = Path(base["save_dir"]) / sweep_name
    results_path = sweep_dir / "sweep.csv"
    rows = []

    for i, (config, overrides) in enumerate(trials):
        out_dir = Path(config["save_dir"]) / config["name"]
        header = f"[{i + 1}/{len(trials)}] {config['name']}  {trial_columns(overrides)}"

        reason = rejection_reason(config)
        if reason:
            print(f"{header}  SKIP ({reason})")
            continue
        if out_dir.exists() and not rerun_existing:
            print(f"{header}  SKIP (output dir exists)")
            continue
        print(header)
        if dry_run:
            continue

        try:
            rows.append(run_trial(config, overrides, sweep_name, wandb_project))
        except Exception as exc:
            traceback.print_exc()
            rows.append(
                {**trial_columns(overrides), "name": config["name"], "error": repr(exc)}
            )
        sweep_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(results_path, index=False)

    if rows:
        table = pd.DataFrame(rows).sort_values("cum_hv", ascending=False)
        print(f"\n=== SWEEP RESULTS (best first, {results_path}) ===")
        print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--sweep_name", type=str, default=None,
        help="output subdirectory and W&B group; defaults to '<config name>_sweep'",
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=None,
        help="override the config's num_timesteps for every trial",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="list the trials without training"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="log each trial to this project"
    )
    parser.add_argument(
        "--rerun_existing", action="store_true",
        help="run trials whose output directory already exists",
    )
    args = parser.parse_args()
    main(
        args.config,
        sweep_name     = args.sweep_name,
        num_timesteps  = args.num_timesteps,
        dry_run        = args.dry_run,
        wandb_project  = args.wandb_project,
        rerun_existing = args.rerun_existing,
    )
