"""Train a multi-objective design hypernetwork on CodesignCheetah.

A multi-objective design hypernetwork produces policy/value networks given both a design
and a tradeoff input: H_{\\pi}(d, w) --> pi_{d,w}(a|s) (and analogously for the value
function). Each epoch trains a grid of sampled designs x sampled tradeoffs.
"""


import argparse

import minimal_mjx as mm
import moplayground as mop

from codesign.envs.CodesignCheetah import CodesignCheetah
from codesign.hyperdesigners import setup_mo_design_hypernetwork

CONFIG_PATH = "config/mo_design_hypernetwork_cheetah.yaml"

def main(config_path: str):
    # (1) Load the config
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    run = mm.utils.logging.initialize_wandb(
        name    = config["name"].replace('/', ''),
        entity  = 'vmadabushi3-georgia-institute-of-technology',
        project = 'codesign-cheetah',
        config  = config
    )

    # Model-as-input env
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env = CodesignCheetah(env_params=env_params, backend="jnp")
    eval_env = CodesignCheetah(env_params=env_params, backend="jnp")

    # Run via minimal-mjx's trainer with our handle_params
    return mm.learning.training.train(
        config,
        env,
        eval_env,
        run=run,
        handle_params=setup_mo_design_hypernetwork,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    args = parser.parse_args()
    main(args.config)
