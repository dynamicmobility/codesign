"""Train a design hypernetwork on MAICheetah.

A design hypernetwork produces policy/value networks given a design input. In
other words, H_{\pi}(d) --> pi_d(a|s) and analagously for the vlaue function.
"""


import argparse

import minimal_mjx as mm
import moplayground as mop

from codesign.envs.MAICheetah import MAICheetah
from codesign.hyperdesigners import setup_design_hypernetwork

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"

def main(config_path: str, run=None):
    # (1) Load the config
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))

    # Model-as-input env
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env = MAICheetah(env_params=env_params, backend="jnp")
    eval_env = MAICheetah(env_params=env_params, backend="jnp")

    # Run via minimal-mjx's trainer with our handle_params
    return mm.learning.training.train(
        config,
        env,
        eval_env,
        run=run,
        handle_params=setup_design_hypernetwork,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    args = parser.parse_args()
    main(args.config)
