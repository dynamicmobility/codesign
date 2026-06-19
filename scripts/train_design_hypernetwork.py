"""Start a `design_hypernetwork` training run on MAICheetah.

Drives the run through minimal-mjx's own trainer, ``mm.learning.training.train``, by
passing our handle_params function explicitly:

  (1) config/design_hypernetwork_cheetah.yaml      — the training config;
  (2) hyperdesigners.setup_design_hypernetwork      — handle_params: config -> (train_fn,
      network_factory).

``mm.learning.training.train`` resolves the algorithm via the ``handle_params`` argument,
so ``design_hypernetwork`` is plugged in WITHOUT being registered in minimal-mjx's
``_ALGO_HANDLERS``. minimal-mjx then owns the output directory, checkpointing
(``save_model``) and progress plotting (``plot_progress``).

    python scripts/train_design_hypernetwork.py
    JAX_PLATFORMS=cpu python scripts/train_design_hypernetwork.py --smoke   # quick CPU run
"""

import argparse

import minimal_mjx as mm
import moplayground as mop

from codesign.envs.MAICheetah import MAICheetah
from codesign.hyperdesigners import setup_design_hypernetwork

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"


def _apply_smoke_overrides(config):
    """Shrink the run to a fast end-to-end check."""
    config["name"] = "test"
    ppo = config["learning_params"]["ppo_params"]
    ppo["num_envs"] = 8
    ppo["batch_size"] = 8
    ppo["num_minibatches"] = 2
    ppo["unroll_length"] = 10
    ppo["episode_length"] = 40
    ppo["num_timesteps"] = 8 * 10 * 2 * 4 * 3  # ~3 epochs
    ppo["num_evals"] = 4
    ppo["num_eval_envs"] = 8


def main(config_path: str, smoke: bool, run=None):
    # (1) Load the config.
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    if smoke:
        _apply_smoke_overrides(config)

    # Model-as-input env (eval_env is only used by mm.train for obs/action sizes).
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env = MAICheetah(env_params=env_params, backend="jnp")
    eval_env = MAICheetah(env_params=env_params, backend="jnp")

    # (2)/(3) Drive the run via minimal-mjx's trainer with our handle_params (unregistered).
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
    parser.add_argument("--smoke", action="store_true", help="quick CPU smoke run")
    args = parser.parse_args()
    main(args.config, args.smoke)
