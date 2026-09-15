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

# Swept params consumed by a derivation instead of being written to a config field.
DERIVED_INPUTS = frozenset({'rollouts_per_step'})

def find_field(config, name, prefix=''):
    """Paths of every field called `name` anywhere in a nested config, spelled the way
    ``mm.flatten_config`` writes them, e.g. 'env_config/codesign/default_design'."""
    paths = []
    for key, value in config.items():
        path = f'{prefix}/{key}' if prefix else key
        if key == name:
            paths.append(path)
        elif hasattr(value, 'items'):
            paths.extend(find_field(value, name, path))
    return paths

def locate(config, param):
    """The ``(container, key)`` a swept param names, or None if the config has no such
    field. A param holding a '/' is a path from the config root; a bare name is searched
    for config-wide and must name exactly one field."""
    path = param
    if '/' not in param:
        matches = find_field(config, param)
        if len(matches) > 1:
            raise ValueError(
                f"Swept param {param!r} names all of {matches}; spell out the one meant."
            )
        if not matches:
            return None
        path = matches[0]

    *groups, key = path.split('/')
    container = config
    for group in groups:
        if not hasattr(container.get(group), 'items'):
            return None
        container = container[group]
    return (container, key) if key in container else None

def swept_fields(sweep_parameters):
    """Leaf field names a sweep sets, however they are spelled (bare, as a path, or inside
    a bundle), so a derivation can test whether a field is swept."""
    fields = set()
    for param, value in sweep_parameters.items():
        if hasattr(value, 'items'):
            fields |= swept_fields(value)
        else:
            fields.add(param.rpartition('/')[2])
    return fields

def write_sweep_params(config, sweep_parameters, prefix=''):
    """Writes each swept param into the field it names. Returns the params naming no
    field, prefixed by the bundle they came from."""
    unmatched = []
    for param, value in sweep_parameters.items():
        target = locate(config, param)
        if hasattr(value, 'items'):
            # A bundle is merged field by field, so the group's unswept fields keep the
            # base config's values. A bundle naming no field is an arm label: its own
            # keys are then resolved against the config.
            group  = config if target is None else target[0][target[1]]
            inner  = prefix if target is None else f'{prefix}{param}/'
            unmatched += write_sweep_params(group, value, inner)
        elif target is None:
            unmatched.append(prefix + param)
        else:
            container, key = target
            container[key] = value
    return unmatched

def apply_sweep_params(config, sweep_parameters):
    """Writes a sweep's sampled params into the training config.

    A param names its field either by a bare name the config defines exactly once
    ('default_design'), or, to disambiguate, by its path from the config root as
    ``mm.flatten_config`` spells it ('env_config/codesign/default_design'). Params are
    restricted to fields the base config already declares, so a typo in one arm is caught
    here rather than silently training the base's setting.
    """
    unmatched = set(write_sweep_params(config, sweep_parameters)) - DERIVED_INPUTS
    if unmatched:
        raise ValueError(
            f"Swept params {sorted(unmatched)} name no field in the config, so they would "
            "be sampled but never applied."
        )

def env_count_key(ppo_params):
    """Name of the env-count field: codesign's trainers call it ``num_parallel_envs``,
    minimal-mjx's ppo ``num_envs``."""
    return 'num_parallel_envs' if 'num_parallel_envs' in ppo_params else 'num_envs'

def derive_batching(ppo_params, sweep_parameters):
    """Derives batch_size so that related args (num_minibatches, etc) are consistent.
    Expects `ppo_params` to already hold this run's swept values.
    Returns the derived values, empty if the sweep doesn't specify batch params.
    """
    BATCHING_PARAMS = (
        'num_envs',
        'num_parallel_envs',
        'num_minibatches',
        'rollouts_per_step',
        'batch_size'
    )
    fields = swept_fields(sweep_parameters)
    if not any(param in fields for param in BATCHING_PARAMS):
        return {}

    if 'batch_size' in fields:
        raise ValueError(
            "batch_size is derived. Sweep 'rollouts_per_step' instead to vary data per training step."
        )

    num_envs        = ppo_params[env_count_key(ppo_params)]
    num_minibatches = ppo_params['num_minibatches']
    rollouts        = sweep_parameters.get('rollouts_per_step', 1)

    # Smallest batch_size making `batch_size * num_minibatches` a multiple of num_envs.
    batch_size = rollouts * num_envs // math.gcd(num_envs, num_minibatches)

    assert batch_size * num_minibatches % num_envs == 0, (num_envs, num_minibatches, batch_size)
    ppo_params['batch_size'] = batch_size
    return {'batch_size': batch_size}

def define_sweep_metric(run, metric):
    """Registers the sweep objective when it names a summary aggregate, e.g.
    'Cumulative Hypervolume.max'. wandb otherwise summarizes a metric by its last
    logged value, which scores a fluctuating objective on wherever it happened to end.
    """
    if metric is None:
        return
    name, _, aggregate = metric['name'].rpartition('.')
    if aggregate in ('min', 'max', 'mean'):
        run.define_metric(name, summary=aggregate)

def derive_eval_grid(learning_params):
    """Derives num_eval_envs so the eval envs split evenly into the
    num_designs x num_tradeoffs cells the design-hypernetwork trainers assert on.
    Returns the derived values, empty if the config has no design grid.
    """
    ppo_params = learning_params['ppo_params']
    design     = learning_params.get('design_sampling')
    tradeoff   = learning_params.get('tradeoff_sampling')
    if design is None:
        return {}

    num_cells = design['num_designs'] * (tradeoff['num_tradeoffs'] if tradeoff else 1)
    key = env_count_key(ppo_params)
    if ppo_params[key] % num_cells != 0:
        raise ValueError(
            f"{key} ({ppo_params[key]}) must be a multiple of "
            f"num_designs * num_tradeoffs ({num_cells})."
        )

    # Smallest multiple of num_cells that is at least the configured num_eval_envs.
    num_eval_envs = math.ceil(ppo_params['num_eval_envs'] / num_cells) * num_cells
    ppo_params['num_eval_envs'] = num_eval_envs
    return {'num_eval_envs': num_eval_envs}

def run_sweep(wandb_sweep_config, codesign_config, PACE=False, count=3):
    """Runs a hyperparameter sweep"""

    def edit_and_train(config=None):
        """Edits a codesign config (specified below via a wandb sweep config)"""
        with wandb.init(config=config) as run:
            define_sweep_metric(run, wandb_sweep_config.get('metric'))

            sweep_parameters        = dict(run.config)
            train_config            = mm.deepcopy_config(codesign_config)
            learning_params         = train_config['learning_params']
            ppo_params              = learning_params['ppo_params']

            apply_sweep_params(train_config, sweep_parameters)

            # Derived after the sweep is applied, so save_dir itself can be swept.
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

            # Derive hyperparameters constrained by the swept ones.
            derived = {
                **derive_batching(ppo_params, sweep_parameters),
                **derive_eval_grid(learning_params),
            }
            run.config.update(derived, allow_val_change=True)

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
        entity=os.environ["WANDB_ENTITY"],
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
