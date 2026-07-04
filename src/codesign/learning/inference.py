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
)
from codesign.hyperdesigners.factory import (
    setup_design_hypernetwork,
    setup_mo_design_hypernetwork,
)


def load_design_networks(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the design-hypernetwork ``DesignHypernetNetworks`` + saved params.

    Rebuilds the networks from the checkpoint's saved ``config.json`` (which stores the
    observation/action sizes and architecture ``network_factory_kwargs``), so no live
    env is needed to recover those sizes. ``design_dim`` and the init ``key`` are bound
    onto the factory here since they're passed at call time during training (and so are
    not captured in the saved kwargs).

    Args:
        config: the run config dict (as written to ``config.yaml`` at train time).
        network_factory: design-network factory; defaults to the one from
            :func:`setup_design_hypernetwork`.
        path: explicit checkpoint dir; defaults to the latest under ``save_dir/name``.
        quiet: if ``False``, print the resolved checkpoint path.

    Returns:
        ``(design_networks, params)`` where ``params = (normalizer_params,
        hypernet_params)``.
    """
    if network_factory is None:
        _, network_factory = setup_design_hypernetwork(config)
    if path is None:
        path = mm.learning.inference.get_last_model(config)
    path = epath.Path(path)
    if not quiet:
        print(f"Loading model at {path.as_posix()}")

    fullpath = path.resolve()
    params_config = checkpoint.load_config(fullpath)
    params = checkpoint.load(fullpath)

    design_dim = int(config["learning_params"]["design_params"]["design_dim"])
    network_factory = functools.partial(
        network_factory,
        design_dim=design_dim,
        key=jax.random.PRNGKey(0),
    )
    design_networks = get_network(params_config, network_factory)
    return design_networks, params


def load_design_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the design-hypernetwork inference fn + saved params from a checkpoint.

    Returns ``(inference_fn, params)`` where ``params = (normalizer_params,
    hypernet_params)`` and ``inference_fn(params, design, deterministic=False)`` yields a
    ``policy(obs, key)``. Mirrors
    ``moplayground.learning.inference.load_hypernetwork_inference_fn``.
    """
    design_networks, params = load_design_networks(
        config, network_factory, path, quiet
    )
    inference_fn = make_design_inference_fn(design_networks)
    return inference_fn, params


def load_mo_design_networks(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the multi-objective design-hypernetwork ``DesignHypernetNetworks`` + params.

    The multi-objective analogue of :func:`load_design_networks`. Rebuilds the ``H(d, w)``
    networks from the checkpoint's saved ``config.json`` architecture kwargs. ``design_dim``,
    ``num_objectives`` and the init ``key`` are passed at call time during training (so they
    are not captured in the saved ``network_factory_kwargs``) and are re-bound here:
    ``design_dim`` from ``design_params`` and ``num_objectives`` from the env's objective
    list in ``env_config.reward.optimization.objectives``.

    Returns ``(design_networks, params)`` where ``params = (normalizer_params,
    hypernet_params)``.
    """
    if network_factory is None:
        _, network_factory = setup_mo_design_hypernetwork(config)
    if path is None:
        path = mm.learning.inference.get_last_model(config)
    path = epath.Path(path)
    if not quiet:
        print(f"Loading model at {path.as_posix()}")

    fullpath = path.resolve()
    params_config = checkpoint.load_config(fullpath)
    params = checkpoint.load(fullpath)

    design_dim = int(config["learning_params"]["design_params"]["design_dim"])
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
    return design_networks, params


def load_mo_design_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the MO design-hypernetwork inference fn + saved params from a checkpoint.

    Returns ``(inference_fn, params)`` where ``params = (normalizer_params,
    hypernet_params)`` and ``inference_fn(params, designs, directives, deterministic=False)``
    yields a ``policy(obs, key)``. The multi-objective analogue of
    :func:`load_design_hypernetwork`.
    """
    design_networks, params = load_mo_design_networks(
        config, network_factory, path, quiet
    )
    inference_fn = make_mo_design_inference_fn(design_networks)
    return inference_fn, params
