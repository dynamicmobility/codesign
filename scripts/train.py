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
        case 'design_hypernetwork':
            return codesign.hyperdesigners.setup_design_hypernetwork
        case 'mo_design_hypernetwork':
            return codesign.hyperdesigners.setup_mo_design_hypernetwork
        case 'ppo':
            return None


def get_progress_fn(config, env: codesign.CodesignBase):
    """Custom progress callback for the MO design hypernetwork (per-design Pareto
    frontiers, one subplot per checkpoint); ``None`` falls back to minimal-mjx's default."""
    
    if config.algorithm == 'mo_design_hypernetwork':
        training_data = codesign.MODesignTrainingPlottingInfo(
            start_time = time.time(),
            labels     = env.objectives,
        )
        # return functools.partial(plot_mo_design_progress, training_data=training_data)
        return functools.partial(codesign.plot_mean_hv_progress, training_data=training_data)
    elif config.algorithm == 'design_hypernetwork' or config.algorithm == 'ppo':
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
        case 'design_hypernetwork':
            env = codesign.CodesignMO2SO(
                env       = env,
                weighting = config.env_config.reward.optimization.default_scalarization
            )
        case 'mo_design_hypernetwork':
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

    if config['algorithm'] == 'ppo' or num_designs < 2:
        designs = [codesign.default_video_design(config)]
    else:
        design_params = config['env_config']['codesign']
        designs = list(codesign.maximin_designs(
            num_designs,
            low  = np.asarray(design_params['low'], np.float64),
            high = np.asarray(design_params['high'], np.float64),
            dim  = len(design_params['low']),
        ))

    # Only the MO hypernetwork takes a tradeoff; (None, None) is one unconditioned pass.
    if config['algorithm'] == 'mo_design_hypernetwork':
        tradeoffs = codesign.extreme_tradeoffs_with_labels(config)
    else:
        tradeoffs = [(None, None)]

    for label, tradeoff in tradeoffs:
        for design in designs:
            d = float(np.asarray(design).reshape(-1)[0])
            # The designs repeat at every corner, so the tradeoff has to be in both names.
            stem    = f'rollout_d{d:.3f}' if label is None else f'rollout_w-{label}_d{d:.3f}'
            log_key = f'rollout/d={d:.3f}' if label is None else f'rollout/{label}/d={d:.3f}'
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


def train(config, log_video=True, **video_kwargs):
    # run = None
    run = mm.utils.logging.initialize_wandb(
        name    = config["name"].replace('/', ''),
        entity  = 'vmadabushi3-georgia-institute-of-technology',
        project = 'codesign',
        config  = config
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
        progress_fn=get_progress_fn(config, env),
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
        num_designs = args.num_designs,
        n_steps     = args.video_steps,
        camera      = args.video_camera,
    )
