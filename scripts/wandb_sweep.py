import os
os.environ.setdefault("MUJOCO_GL", "egl")

import minimal_mjx as mm
import wandb
import scripts.train as train
import argparse
import math
from pathlib import Path
import os
import codesign

class UniqueSet(set):
    def add(self, element):
        if element in self:
            raise ValueError(f"Duplicate item added: {element}")
        super().add(element)

def derive_batching(ppo_params, sweep_parameters):
    """Derives batch_size so that related args (num_minibatches, etc) are consistent.
    Returns the derived values, or None if the sweep doesn't specify batch params.
    """
    BATCHING_PARAMS = (
        'num_envs', 
        'num_minibatches', 
        'rollouts_per_step', 
        'batch_size'
    )
    if not any(param in sweep_parameters for param in BATCHING_PARAMS):
        return None

    if 'batch_size' in sweep_parameters:
        raise ValueError(
            "batch_size is derived. Sweep 'rollouts_per_step' instead to vary data per training step."
        )

    num_envs        = sweep_parameters.get('num_envs', ppo_params['num_envs'])
    num_minibatches = sweep_parameters.get('num_minibatches', ppo_params['num_minibatches'])
    rollouts        = sweep_parameters.get('rollouts_per_step', 1)

    # Smallest batch_size making `batch_size * num_minibatches` a multiple of num_envs.
    batch_size = rollouts * num_envs // math.gcd(num_envs, num_minibatches)

    derived = {
        'num_envs'        : num_envs,
        'num_minibatches' : num_minibatches,
        'batch_size'      : batch_size,
    }
    assert batch_size * num_minibatches % num_envs == 0, derived
    ppo_params.update(derived)
    return derived

def run_sweep(wandb_sweep_config, codesign_config, PACE=False, count=3):
    """Runs a hyperparameter sweep"""

    def edit_and_train(config=None):
        """Edits a codesign config (specified below via a wandb sweep config)"""
        with wandb.init(config=config) as run:
            sweep_parameters        = dict(run.config)
            train_config            = mm.deepcopy_config(codesign_config)
            learning_params         = train_config['learning_params']
            ppo_params              = learning_params['ppo_params']
            network_params          = learning_params['network_params']
            
            train_config['save_dir'] = (Path(train_config['save_dir']) / str(run.id)).as_posix()
            if PACE:                
                save_rel_path = Path(train_config['save_dir'])
                scratch_rel_path = Path('scratch/logs/codesign')
                home_path = Path.home()
                
                save_path = home_path / scratch_rel_path / save_rel_path
                print(save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    print(f"Directory '{save_path}' created.")
                else:
                    print(f"Directory '{save_path}' already exists.")
                
                train_config['save_dir'] = save_path.as_posix()
            
            # derive hyperparameters that have constraints. Only batch_size is written
            derived = derive_batching(ppo_params, sweep_parameters)
            if derived is not None:
                run.config.update({'batch_size': derived['batch_size']}, allow_val_change=True)

            # Update config with sweep instance params. Error if duplicate is found
            used_params = UniqueSet()
            for param in sweep_parameters.keys():
                
                if param in ppo_params:
                    ppo_params[param] = sweep_parameters[param]
                    used_params.add(param)
                
                if param in network_params:
                    network_params[param] = sweep_parameters[param]
                    used_params.add(param)

            # Publish all of train config, not just the swept params, so we
            # can rebuild the whole config from the run later.
            run.config.update(mm.flatten_config(train_config), allow_val_change=True)

            # Codesign Env
            env, _        = codesign.load_env(train_config)
            eval_env, _   = codesign.load_env(train_config)
            
            env           = train.wrap_env(train_config, env)
            eval_env      = train.wrap_env(train_config, eval_env)

            setup_fn      = train.get_handle_params(train_config)

            # Run via minimal-mjx's trainer with our handle_params
            ret = mm.learning.training.train(
                config          = train_config,
                env             = env,
                eval_env        = eval_env,
                run             = run,
                handle_params   = setup_fn,
                progress_fn     = train.get_progress_fn(train_config, env),
            )
            train.log_rollout_videos(train_config, run=run)
            # try:
            #     train.log_rollout_videos(train_config, run=run)
            # except Exception as e:
            #     print(f"Failed to log rollout video: {e!r}")
    
    sweep_config = wandb_sweep_config.to_dict()
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity='vmadabushi3-georgia-institute-of-technology',
        project="codesign"
    )
    wandb.agent(sweep_id, edit_and_train, count=count)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str)
    parser.add_argument("--sweep", type=str)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--pace", type=bool, default=False)
    args = parser.parse_args()
    config = mm.read_config(args.config)
    sweep = mm.read_config(args.sweep)
    run_sweep(sweep, config, count=args.num_trials, PACE=args.pace)
