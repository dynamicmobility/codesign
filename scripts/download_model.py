"""Download a trained policy and its full config from a W&B run.
"""

import argparse
from pathlib import Path

import minimal_mjx as mm
import wandb

ENTITY  = 'vmadabushi3-georgia-institute-of-technology'
PROJECT = 'codesign'


def download_model(run_id, save_dir, name=None, entity=ENTITY, project=PROJECT,
                   artifact_prefix='hypernetworks'):
    """Download run `run_id`'s newest checkpoint plus its config into `save_dir/name`.
    """
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_id}')

    name = name or run_id
    output_dir = Path(save_dir)
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = mm.unflatten_config(run.config)
    config['save_dir'] = output_dir.as_posix()
    config['name'] = str(name)
    mm.save_config(config, run_dir / 'config.yaml')

    artifact = mm.get_latest_artifact(run, artifact_prefix)
    iteration = artifact.metadata.get('iteration')
    if iteration is None:
        raise ValueError(
            f"Artifact '{artifact.name}' has no 'iteration' metadata; cannot name its "
            f"checkpoint directory (loading expects step-numbered directories)."
        )
    # Checkpoints must land in a step-numbered directory -- get_all_models() ignores the rest.
    artifact.download(root=str(run_dir / str(iteration)))

    return config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_id', type=str, required=True,
                        help='W&B run id to download (e.g. the 8-character run id)')
    parser.add_argument('--save_dir', type=str, default='results/wandb-downloads',
                        help='Directory to download into (default: results/wandb-downloads)')
    parser.add_argument('--name', type=str, default=None,
                        help='Subdirectory name for the run (default: the run id)')
    parser.add_argument('--entity', type=str, default=ENTITY)
    parser.add_argument('--project', type=str, default=PROJECT)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or args.run_id

    print(f"Downloading run {args.entity}/{args.project}/{args.run_id}...")
    config = download_model(
        run_id     = args.run_id,
        save_dir   = args.save_dir,
        name       = name,
        entity     = args.entity,
        project    = args.project,
    )

    run_dir = Path(args.save_dir) / name
    print(f"Done! algorithm={config.get('algorithm')} env={config.get('env_name')}")
    print(f"Config saved to: {run_dir / 'config.yaml'}")


if __name__ == '__main__':
    main()
