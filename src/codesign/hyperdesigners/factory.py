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
    """
    lp = config["learning_params"]
    ppo = dict(lp["ppo_params"])
    net = dict(lp["network_params"])
    design = dict(config['env_config']['codesign'])

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
        design_low        = design["low"],
        design_high       = design["high"],
        design_dim        = len(design['low']),
        **ppo,
    )
    return train_fn, network_factory


def setup_mo_design_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the multi-objective design hypernetwork.
    """
    lp                = config["learning_params"]
    ppo               = dict(lp["ppo_params"])
    net               = dict(lp["network_params"])
    codesign          = dict(config['env_config']['codesign'])
    design_sampling   = dict(lp['design_sampling'])
    tradeoff_sampling = dict(lp['tradeoff_sampling'])

    network_factory = functools.partial(
        make_mo_design_hypernet_networks,
        hypersize                   = tuple(net["hypersize"]),
        num_features                = net["num_features"],
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
    )

    train_fn = functools.partial(
        train_mo_design_hypernetwork,
        network_factory       = network_factory,
        design_low            = codesign["low"],
        design_high           = codesign["high"],
        design_dim            = len(codesign["low"]),
        num_designs           = design_sampling["num_designs"],
        resamples_per_epoch   = design_sampling["resamples_per_epoch"],
        num_tradeoffs         = tradeoff_sampling["num_tradeoffs"],
        alpha                 = tradeoff_sampling["alpha"],
        sampling              = tradeoff_sampling["sampling"],
        **ppo,
    )
    return train_fn, network_factory
