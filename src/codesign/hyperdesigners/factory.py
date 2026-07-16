import functools

from codesign.hyperdesigners.design_hypernetwork import train_design_hypernetwork
from codesign.hyperdesigners.mo_design_hypernetwork import (
    train_mo_design_hypernetwork,
)
from codesign.hyperdesigners.networks import (
    make_design_hypernet_networks,
    make_mo_design_hypernet_networks,
)


def setup_design_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the design hypernetwork.

    ``network_factory`` is a ``functools.partial`` of
    :func:`make_design_hypernet_networks` with the architecture (hypernet size, feature
    count, target policy/value MLP sizes) pre-bound; ``train_fn`` is a
    ``functools.partial`` of :func:`train_design_hypernetwork` with all algorithm and
    design hyperparameters pre-bound. ``train_fn`` still expects ``environment`` (and,
    optionally, ``progress_fn`` / ``policy_params_fn``) at call time.
    """
    lp = config["learning_params"]
    ppo = dict(lp["ppo_params"])
    net = dict(lp["design_hypernetwork_params"])
    design = dict(lp["design_params"])

    network_factory = functools.partial(
        make_design_hypernet_networks,
        hypersize                   = tuple(net["hypersize"]),
        num_features                = net["num_features"],
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
    )

    train_fn = functools.partial(
        train_design_hypernetwork,
        network_factory   = network_factory,
        design_low        = design["design_low"],
        design_high       = design["design_high"],
        design_dim        = design["design_dim"],
        **({"reward_objective_weights": tuple(lp["reward_objective_weights"])}
           if "reward_objective_weights" in lp else {}),
        **ppo,
    )
    return train_fn, network_factory


def setup_mo_design_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the multi-objective design hypernetwork.

    Mirrors :func:`setup_design_hypernetwork` but binds
    :func:`make_mo_design_hypernet_networks` and :func:`train_mo_design_hypernetwork`, and
    reads the design x tradeoff grid / tradeoff-sampling knobs from
    ``learning_params.tradeoff_params``.
    """
    lp        = config["learning_params"]
    ppo       = dict(lp["ppo_params"])
    net       = dict(lp["network_params"])
    design    = dict(lp["design_params"])
    tradeoff  = dict(lp["tradeoff_params"])

    network_factory = functools.partial(
        make_mo_design_hypernet_networks,
        hypersize                   = tuple(net["hypersize"]),
        num_features                = net["num_features"],
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
    )

    train_fn = functools.partial(
        train_mo_design_hypernetwork,
        network_factory   = network_factory,
        design_low        = design["design_low"],
        design_high       = design["design_high"],
        design_dim        = design["design_dim"],
        num_designs       = design["num_designs"],
        num_tradeoffs     = tradeoff["num_tradeoffs"],
        alpha             = tradeoff["alpha"],
        sampling          = tradeoff["sampling"],
        warmup_frac       = tradeoff["warmup_frac"],
        **ppo,
    )
    return train_fn, network_factory
