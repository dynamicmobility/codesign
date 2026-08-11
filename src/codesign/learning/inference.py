"""Reconstruct trained policies from checkpoints for evaluation/rollout.

Shared loaders so rollout/eval scripts don't each re-implement checkpoint discovery
and network reconstruction. The networks are rebuilt from the ``config.json`` brax
writes alongside the params (observation/action sizes + ``network_factory_kwargs``),
so callers don't need a live env to recover obs/action sizes — mirroring
``moplayground.learning.inference.load_hypernetworks``.
"""

import functools

import jax
from etils import epath
from brax.training.checkpoint import get_network
from brax.training.agents.ppo import checkpoint
import minimal_mjx as mm

from codesign.hyperdesigners.networks import (
    make_design_inference_fn,
    make_mo_design_inference_fn,
    make_design_predictor_inference_fn,
    make_value_fn
)
from codesign.hyperdesigners.factory import (
    setup_design_hypernetwork,
    setup_mo_design_hypernetwork,
    setup_mo_design_predictor_hypernetwork,
)


def _load_checkpoint(config, path, quiet):
    """Resolve the checkpoint dir and load its saved network config + params.

    Args:
        config: the run config dict (as written to ``config.yaml`` at train time).
        path: explicit checkpoint dir; defaults to the latest under ``save_dir/name``.
        quiet: if ``False``, print the resolved checkpoint path.

    Returns:
        ``(params_config, params)`` where ``params = (normalizer_params, hypernet_params)``.
    """
    if path is None:
        path = mm.learning.inference.get_last_model(config)
    path = epath.Path(path)
    if not quiet:
        print(f"Loading model at {path.as_posix()}")

    fullpath = path.resolve()
    return checkpoint.load_config(fullpath), checkpoint.load(fullpath)


def load_design_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the design-hypernetwork inference fn + saved params from a checkpoint.
    """
    if network_factory is None:
        _, network_factory = setup_design_hypernetwork(config)
    params_config, params = _load_checkpoint(config, path, quiet)

    design_dim = len(config["env_config"]["codesign"]["low"])
    network_factory = functools.partial(
        network_factory,
        design_dim=design_dim,
        key=jax.random.PRNGKey(0),
    )
    design_networks = get_network(params_config, network_factory)
    inference_fn = make_design_inference_fn(design_networks)
    return inference_fn, params

def load_design_value_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the design-hypernetwork inference fn + saved params from a checkpoint.
    """
    if network_factory is None:
        _, network_factory = setup_design_hypernetwork(config)
    params_config, params = _load_checkpoint(config, path, quiet)

    design_dim = len(config["env_config"]["codesign"]["low"])
    network_factory = functools.partial(
        network_factory,
        design_dim=design_dim,
        key=jax.random.PRNGKey(0),
    )
    design_networks = get_network(params_config, network_factory)
    inference_fn = make_value_fn(design_networks)
    return inference_fn, params

def load_mo_design_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the MO design-hypernetwork inference fn + saved params from a checkpoint.
    """
    if network_factory is None:
        _, network_factory = setup_mo_design_hypernetwork(config)
    params_config, params = _load_checkpoint(config, path, quiet)

    design_dim = len(config["env_config"]["codesign"]["low"])
    num_objectives = len(
        config["env_config"]["reward"]["optimization"]["objectives"]
    )
    network_factory = functools.partial(
        network_factory,
        design_dim=design_dim,
        num_objectives=num_objectives,
        key=jax.random.PRNGKey(0),
    )
    design_networks = get_network(params_config, network_factory)
    inference_fn = make_mo_design_inference_fn(design_networks)
    return inference_fn, params


def load_mo_design_predictor_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the MO design predictor hypernetwork from a checkpoint.

    Returns ``(inference_fn, design_predictor_inference_fn, params)`` where ``params`` is
    ``(normalizer_params, hypernet_params, design_predictor_params)``.
    """
    if network_factory is None:
        _, network_factory = setup_mo_design_predictor_hypernetwork(config)
    params_config, params = _load_checkpoint(config, path, quiet)

    design_dim = len(config["env_config"]["codesign"]["low"])
    num_objectives = len(
        config["env_config"]["reward"]["optimization"]["objectives"]
    )
    network_factory = functools.partial(
        network_factory,
        design_dim=design_dim,
        num_objectives=num_objectives,
        key=jax.random.PRNGKey(0),
    )
    design_networks = get_network(params_config, network_factory)
    return (
        make_mo_design_inference_fn(design_networks),
        make_design_predictor_inference_fn(design_networks),
        params,
    )
