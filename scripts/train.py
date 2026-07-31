"""Generic training script for hyperdesigners"""

import argparse
import functools
import time

import minimal_mjx as mm
import moplayground as mop

from codesign.envs.CodesignBase import MOCodesignBase
from codesign.envs.EnvLoader import load_env
from codesign.utils.plotting import (
    MODesignTrainingPlottingInfo,
    plot_mo_design_progress,
)
import codesign

def get_handle_params(config):
    match config.algorithm:
        case 'design_hypernetwork':
            return codesign.hyperdesigners.setup_design_hypernetwork
        case 'mo_design_hypernetwork':
            return codesign.hyperdesigners.setup_mo_design_hypernetwork
        case 'ppo':
            return None


def get_progress_fn(config, env: MOCodesignBase):
    """Custom progress callback for the MO design hypernetwork (per-design Pareto
    frontiers, one subplot per checkpoint); ``None`` falls back to minimal-mjx's default."""
    
    if config.algorithm == 'mo_design_hypernetwork':
        training_data = MODesignTrainingPlottingInfo(
            start_time = time.time(),
            labels     = env.objectives,
        )
        # return functools.partial(plot_mo_design_progress, training_data=training_data)
        return functools.partial(codesign.plot_cum_hv_progress, training_data=training_data)
    elif config.algorithm == 'design_hypernetwork' or config.algorithm == 'ppo':
        return None
    else:
        raise Exception(f'Unknown algorithm {config.algorithm}')
    

def wrap_env(config, env):
    match config.algorithm:
        case 'ppo':
            env = codesign.CodesignMO2SO(
                env       = env,
                weighting = config.learning_params.reward_objective_weights
            )
            env = codesign.Codesign2SingleDesign(
                env = env,
                design = config.learning_params.default_design
            )
        case 'design_hypernetwork':
            env = codesign.CodesignMO2SO(
                env       = env,
                weighting = config.learning_params.reward_objective_weights
            )
        case 'mo_design_hypernetwork':
            pass
        case e:
            raise Exception(f'Unknown algorithm {e}')
    
    return env


def train(config):    
    # run = None
    run = mm.utils.logging.initialize_wandb(
        name    = config["name"].replace('/', ''),
        entity  = 'vmadabushi3-georgia-institute-of-technology',
        project = 'codesign',
        config  = config
    )

    # Codesign Env
    env, _ = load_env(config)
    eval_env, _ = load_env(config)
    
    env = wrap_env(config, env)
    eval_env = wrap_env(config, eval_env)

    setup_fn = get_handle_params(config)

    # Run via minimal-mjx's trainer with our handle_params
    return mm.learning.training.train(
        config,
        env,
        eval_env,
        run=run,
        handle_params=setup_fn,
        progress_fn=get_progress_fn(config, env),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    config = mm.read_config(args.config)
    train(config)
