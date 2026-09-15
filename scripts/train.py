"""Generic training script for hyperdesigners"""

import os

# Headless (GPU) rendering for the end-of-training rollout video; set before mujoco loads.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import time
from pathlib import Path

import minimal_mjx as mm
import moplayground as mop
import numpy as np
import codesign

def get_handle_params(config):
    match config.algorithm:
        case 'design_mlp':
            return codesign.hyperdesigners.setup_design_mlp
        case 'mo_design_mlp':
            return codesign.hyperdesigners.setup_mo_design_mlp
        case 'design_hypernetwork':
            return codesign.hyperdesigners.setup_design_hypernetwork
        case 'design_lookup_hypernetwork':
            return codesign.hyperdesigners.setup_design_lookup_hypernetwork
        case 'mo_design_hypernetwork':
            return codesign.hyperdesigners.setup_mo_design_hypernetwork
        case 'mo_design_predictor_hypernetwork':
            return codesign.hyperdesigners.setup_mo_design_predictor_hypernetwork
        case 'morlax':
            return setup_morlax
        case 'ppo':
            return None


def setup_morlax(config):
    """``mop.setup_morlax`` pinned to moplayground's multi-objective env wrapper.

    minimal-mjx hands every algorithm brax's single-objective wrapper, whose episode
    bookkeeping sums a scalar reward per env; morlax's envs return one reward per
    objective, so the training and eval envs need ``mop.mo_wrapper`` instead.
    """
    train_fn, network_factory = mop.setup_morlax(config)

    def mo_train_fn(*args, wrap_env_fn=None, **kwargs):
        return train_fn(*args, wrap_env_fn=mop.mo_wrapper, **kwargs)

    return mo_train_fn, network_factory


def pareto_aux(grid, ref_point=None): # TODO: move to plotting
    """The hypervolume and spacing ``plot_design_pareto_progress`` records per eval."""
    mean_rewards = grid.mean_rewards
    hv, sp = mop.get_pareto_statistics(
        mean_rewards.reshape(-1, mean_rewards.shape[-1]), ref_point=ref_point
    )
    return {'hv': float(hv), 'sp': float(sp)}


def get_progress_fn(config, env: codesign.CodesignBase, resume=False):
    """Custom progress callback for the MO design hypernetwork (per-design Pareto
    frontiers, one subplot per checkpoint); ``None`` falls back to minimal-mjx's default.

    ``resume`` reloads the evals already in the run directory, so a continued run's csv
    and figures cover the whole run rather than only the epochs this job adds."""

    run_dir = Path(config['save_dir']) / config['name']
    # Evals at or past the restart point get redone, so they are not read back.
    resumed = mm.find_resume(run_dir) if resume else None
    before = None if resumed is None else resumed.step

    if config.algorithm in ('mo_design_hypernetwork', 'mo_design_predictor_hypernetwork', 'mo_design_mlp'):
        optimization = config.env_config.reward.optimization
        ref_point = optimization.get('reference_point', None)
        training_data = codesign.MODesignTrainingPlottingInfo(
            start_time = time.time(),
            labels     = env.objectives,
            ref_point  = ref_point,
        )
        if resumed is not None:
            codesign.load_training_data(
                training_data, run_dir, 'design_pareto_progress.csv',
                # hv and sp are not in the csv, but they are functions of the grid.
                aux_fn=functools.partial(pareto_aux, ref_point=ref_point),
                before=before,
            )
        return functools.partial(codesign.plot_design_pareto_progress, training_data=training_data)
    
    elif config.algorithm == 'design_lookup_hypernetwork' or (
        config.algorithm == 'design_hypernetwork'
        and config.learning_params.get('design_sampling', {}).get('strategy')
        in ('fixed', 'noisy-fixed')
    ):
        # save to designs.csv
        training_data = codesign.MODesignTrainingPlottingInfo(
            start_time = time.time(),
            labels     = getattr(env, 'objectives', None) or [],
        )
        if resumed is not None:
            codesign.load_training_data(
                training_data, run_dir, 'design_rewards_progress.csv',
                aux_grids={'anchor_grid': 'anchor_eval_grid'},
                before=before,
            )
        return functools.partial(
            codesign.plot_design_rewards_progress, training_data=training_data
        )
    elif config.algorithm == 'morlax':
        training_data = mop.MOTrainingPlottingInfo(
            start_time = time.time(),
            labels     = env.objectives,
        )
        def morlax_progress(num_steps, metrics, times, save_dir, run=None, **kwargs):
            """``mop.plot_mo_progress`` under minimal-mjx's callback signature; the
            scalar-reward curve arguments it passes go unused."""
            times.append(time.time())
            mop.plot_mo_progress(num_steps, metrics, training_data, save_dir, run)
        return morlax_progress
    elif config.algorithm in ('design_hypernetwork', 'ppo', 'design_mlp'):
        return None
    else:
        raise Exception(f'Unknown algorithm {config.algorithm}')


def wrap_env(config, env):
    match config.algorithm:
        case 'ppo':
            env = codesign.CodesignMO2SO(
                env       = env,
                weighting = config.env_config.reward.optimization.default_scalarization
            )
            env = codesign.Codesign2SingleDesign(
                env = env,
                design = config.env_config.codesign.default_design
            )
        case 'design_hypernetwork' | 'design_lookup_hypernetwork' | 'design_mlp':
            env = codesign.CodesignMO2SO(
                env       = env,
                weighting = config.env_config.reward.optimization.default_scalarization
            )
        case 'morlax':
            env = codesign.Codesign2SingleDesign(
                env = env,
                design = config.env_config.codesign.default_design
            )
        case 'mo_design_hypernetwork' | 'mo_design_predictor_hypernetwork' | 'mo_design_mlp':
            pass
        case e:
            raise Exception(f'Unknown algorithm {e}')
    
    return env


def log_rollout_videos(
    config,
    run          = None,
    num_designs  = 3,
    n_steps      = 500,
    camera       = 'track',
    seed         = 0,
):
    """Render the trained policy from the latest checkpoint and log it to W&B.
    """
    video_dir = Path(config['save_dir']) / config['name'] / 'videos'
    # Rendering needs a host-side (mujoco, not mjx) env; reused across designs.
    env, _ = codesign.load_env(config, backend='np')

    # The design predictor chooses its own design, so sweeping designs would just repeat.
    predicts_design = config['algorithm'] == 'mo_design_predictor_hypernetwork'
    if config['algorithm'] == 'ppo' or predicts_design or num_designs < 2:
        designs = [codesign.default_video_design(config)]
    else:
        design_params = config['env_config']['codesign']
        designs = list(codesign.maximin_designs(
            num_designs,
            low  = np.asarray(design_params['low'], np.float64),
            high = np.asarray(design_params['high'], np.float64),
            dim  = len(design_params['low']),
        ))

    # Only the MO hypernetworks take a tradeoff; (None, None) is one unconditioned pass.
    if config['algorithm'] in ('mo_design_hypernetwork', 'mo_design_predictor_hypernetwork'):
        tradeoffs = codesign.extreme_tradeoffs_with_labels(config)
    else:
        tradeoffs = [(None, None)]

    for label, tradeoff in tradeoffs:
        for design in designs:
            d = float(np.asarray(design).reshape(-1)[0])
            # The designs repeat at every corner, so the tradeoff has to be in both names.
            if predicts_design:
                stem, log_key = f'rollout_w-{label}', f'rollout/{label}'
            elif label is None:
                stem, log_key = f'rollout_d{d:.3f}', f'rollout/d={d:.3f}'
            else:
                stem, log_key = f'rollout_w-{label}_d{d:.3f}', f'rollout/{label}/d={d:.3f}'
            codesign.save_policy_rollout_video(
                config,
                video_dir / f'{stem}.mp4',
                env      = env,
                design   = design,
                tradeoff = tradeoff,
                n_steps  = n_steps,
                camera   = camera,
                seed     = seed,
                run      = run,
                log_key  = log_key,
            )


def train(config, log_video=True, resume=False, **video_kwargs):
    # A continued run keeps logging to the W&B run that opened its directory.
    run_id = mm.utils.logging.load_run_id(
        Path(config['save_dir']) / config['name']
    ) if resume else None
    # run = None
    run = mm.utils.logging.initialize_wandb(
        name    = config["name"].replace('/', ''),
        entity  = os.environ["WANDB_ENTITY"],
        project = 'codesign',
        config  = config,
        id      = run_id,
        resume  = 'allow' if run_id else None,
    )

    # Codesign Env
    env, _ = codesign.load_env(config)
    eval_env, _ = codesign.load_env(config)
    
    env = wrap_env(config, env)
    eval_env = wrap_env(config, eval_env)

    setup_fn = get_handle_params(config)

    # Run via minimal-mjx's trainer with our handle_params
    results = mm.learning.training.train(
        config,
        env,
        eval_env,
        run=run,
        handle_params=setup_fn,
        progress_fn=get_progress_fn(config, env, resume=resume),
        resume=resume,
    )

    if log_video:
        try:
            log_rollout_videos(config, run=run, **video_kwargs)
        except Exception as e:
            print(f"Failed to log rollout video: {e!r}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str)
    parser.add_argument(
        "--no-video", action="store_true",
        help="skip rendering/logging the policy rollout video after training",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="continue the checkpoints already in the run directory instead of "
             "refusing to overwrite it; num_timesteps is then the run's total budget "
             "and only what is left of it gets trained",
    )
    parser.add_argument(
        "--num-designs", type=int, default=3,
        help="designs to sweep across the design range (hypernetworks only); one video each, "
             "or one per extreme tradeoff each for mo_design_hypernetwork",
    )
    parser.add_argument("--video-steps", type=int, default=500, help="rollout length (env steps)")
    parser.add_argument("--video-camera", type=str, default="track", help="render camera name")
    args = parser.parse_args()
    config = mm.read_config(args.config)
    train(
        config,
        log_video   = not args.no_video,
        resume      = args.resume,
        num_designs = args.num_designs,
        n_steps     = args.video_steps,
        camera      = args.video_camera,
    )
