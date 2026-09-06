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

from codesign.hyperdesigners.networks import make_value_fn
from codesign.hyperdesigners.variants.design_hypernetwork import (
    make_design_inference_fn,
    setup_design_hypernetwork,
)
from codesign.hyperdesigners.hypernetworks import sobol_design_table
from codesign.hyperdesigners.variants.design_lookup_hypernetwork import (
    setup_design_lookup_hypernetwork,
    warn_off_table,
)
from codesign.hyperdesigners.variants.mo_design_hypernetwork import (
    make_mo_design_inference_fn,
    setup_mo_design_hypernetwork,
)
from codesign.hyperdesigners.variants.mo_design_predictor_hypernetwork import (
    make_design_predictor_inference_fn,
    setup_mo_design_predictor_hypernetwork,
)
from codesign.hyperdesigners.variants.design_mlp import (
    augment_observation_size,
    setup_design_mlp,
    make_design_mlp_inference_fn
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


def load_design_features(config, path=None, quiet=True):
    """``(features_fn, params)`` for the design hypernetworks, else ``(None, None)``.

    ``features_fn(hypernet_params, normalized_design) -> (batch, num_features)`` is the row
    that multiplies ``W`` in ``flat(d) = features(d) @ W + b``, so it says which experts the
    policy for ``d`` is built from: one-hot for the lookup, whatever the MLP learned for the
    full hypernetwork.
    """
    setup_fn = {
        "design_hypernetwork": setup_design_hypernetwork,
        "design_lookup_hypernetwork": setup_design_lookup_hypernetwork,
    }.get(config["algorithm"])
    if setup_fn is None:
        return None, None

    _, network_factory = setup_fn(config)
    params_config, params = _load_checkpoint(config, path, quiet)
    network_factory = functools.partial(
        network_factory,
        design_dim=len(config["env_config"]["codesign"]["low"]),
        key=jax.random.PRNGKey(0),
    )
    return get_network(params_config, network_factory).hypernetwork.features, params


def load_design_lookup_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the lookup (warm-up) hypernetwork's inference fn + saved params.

    ``(num_designs, design_seed, design_low, design_high)`` ride along in the checkpoint's
    network config, which fixes the Sobol draw exactly, so only the design width and a
    dummy key are supplied here.
    """
    if network_factory is None:
        _, network_factory = setup_design_lookup_hypernetwork(config)
    params_config, params = _load_checkpoint(config, path, quiet)

    network_factory = functools.partial(
        network_factory,
        design_dim=len(config["env_config"]["codesign"]["low"]),
        key=jax.random.PRNGKey(0),
    )
    design_networks = get_network(params_config, network_factory)
    inference_fn = make_design_inference_fn(design_networks)

    # The lookup snaps a design it does not hold to the nearest one it does, so say so
    # rather than hand back a neighbour's policy for a robot it never saw.
    anchors = params_config.network_factory_kwargs
    table = sobol_design_table(
        anchors["design_seed"], anchors["num_designs"],
        anchors["design_low"], anchors["design_high"],
    )

    def checked_inference_fn(params, design, deterministic: bool = False):
        warn_off_table(design, table)
        return inference_fn(params, design, deterministic=deterministic)

    return checked_inference_fn, params


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


def load_mo_design_value_hypernetwork(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load the value inference function from an MO design-hypernetwork checkpoint."""
    if network_factory is None:
        setup_fn = (
            setup_mo_design_predictor_hypernetwork
            if config["algorithm"] == "mo_design_predictor_hypernetwork"
            else setup_mo_design_hypernetwork
        )
        _, network_factory = setup_fn(config)
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
    return make_value_fn(design_networks), params


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


def load_design_mlp(
    config,
    network_factory=None,
    path=None,
    quiet=True,
):
    """Load a design-conditioned MLP and its saved parameters from a checkpoint."""
    if network_factory is None:
        _, network_factory = setup_design_mlp(config)
    params_config, params = _load_checkpoint(config, path, quiet)

    design_dim = len(config["env_config"]["codesign"]["low"])

    # The checkpoint records the environment's raw observation size, while design_mlp
    # was trained on [observation, design]. Reapply the same widening used in training
    # before Brax reconstructs the policy and value networks.
    def augmented_network_factory(observation_size, action_size, **kwargs):
        return network_factory(
            observation_size=augment_observation_size(
                observation_size, design_dim
            ),
            action_size=action_size,
            design_dim=design_dim,
            key=jax.random.PRNGKey(0),
            **kwargs,
        )

    design_networks = get_network(params_config, augmented_network_factory)
    inference_fn = make_design_mlp_inference_fn(design_networks)
    return inference_fn, params
