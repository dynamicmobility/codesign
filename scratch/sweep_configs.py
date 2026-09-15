"""Print one config attribute for every run in a W&B sweep.

Configs are logged flattened (see `mm.flatten_config`), so nested attributes are
addressed by their `/`-joined path, e.g. 'learning_params/ppo_params/num_minibatches'.
"""

import argparse

import wandb
import os

ENTITY  = os.environ["WANDB_ENTITY"]
PROJECT = 'codesign'

MISSING = object()


def sweep_attributes(sweep_id, attr, entity=ENTITY, project=PROJECT):
    """Yield (run, value of `attr` in run.config) for each run in `sweep_id`,
    with `MISSING` for runs whose config lacks the attribute."""
    api   = wandb.Api()
    sweep = api.sweep(f'{entity}/{project}/{sweep_id}')
    for run in sweep.runs:
        yield run, run.config.get(attr, MISSING)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep_id', type=str, required=True,
                        help='W&B sweep id (the 8-character id, not the full path)')
    parser.add_argument('--attr', type=str, default='name',
                        help="Config key to print, '/'-joined for nested keys (default: name)")
    parser.add_argument('--entity', type=str, default=ENTITY)
    parser.add_argument('--project', type=str, default=PROJECT)
    return parser.parse_args()


def main():
    args = parse_args()
    for run, value in sweep_attributes(args.sweep_id, args.attr, args.entity, args.project):
        shown = '<missing>' if value is MISSING else repr(value)
        print(f'{run.id}  {run.state:<10}  {args.attr}={shown}')


if __name__ == '__main__':
    main()
