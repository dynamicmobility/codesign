import functools

from codesign.hyperdesigners.design_hypernetwork import train_design_hypernetwork
from codesign.hyperdesigners.mo_design_hypernetwork import (
    train_mo_design_hypernetwork,
)
from codesign.hyperdesigners.mo_design_predictor_hypernetwork import (
    train_mo_design_predictor,
)
from codesign.hyperdesigners.networks import (
    make_design_hypernet_networks,
    make_mo_design_hypernet_networks,
    make_mo_design_predictor_hypernet_networks,
)


def setup_design_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the design hypernetwork.
    """
    lp = config["learning_params"]
    ppo = dict(lp["ppo_params"])
    net = dict(lp["network_params"])
    design = dict(config['env_config']['codesign'])
    design_sampling = dict(lp.get("design_sampling", {}))

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
        num_designs       = design_sampling.get("num_designs", 8),
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


def setup_mo_design_predictor_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the MO design predictor hypernetwork.

    ``num_designs`` is the GRPO group size: the number of designs drawn from ``f(d | w)``
    for each tradeoff.
    """
    lp                = config["learning_params"]
    ppo               = dict(lp["ppo_params"])
    net               = dict(lp["network_params"])
    codesign          = dict(config['env_config']['codesign'])
    design_sampling   = dict(lp['design_sampling'])
    tradeoff_sampling = dict(lp['tradeoff_sampling'])
    predictor         = dict(lp['design_predictor'])

    # Network kwargs must carry defaults on the factory so brax's checkpoint config
    # records them and get_network can rebuild the bundle at load time.
    network_factory = functools.partial(
        make_mo_design_predictor_hypernet_networks,
        hypersize                   = tuple(net["hypersize"]),
        num_features                = net["num_features"],
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
        design_hidden_layer_sizes   = tuple(predictor["hidden_layer_sizes"]),
    )

    OPTIONAL_ARGS = ( # TODO: move these into the required args when mature
        (predictor,         "design_learning_rate"),
        (predictor,         "design_entropy_cost"),
        (predictor,         "design_clipping_epsilon"),
        (predictor,         "num_design_updates_per_batch"),
        (predictor,         "num_warmup_iters"),
        (design_sampling,   "num_eval_designs"),
        (tradeoff_sampling, "num_eval_tradeoffs"),
    )
    optional_params = {a: group[a] for group, a in OPTIONAL_ARGS if a in group}

    train_fn = functools.partial(
        train_mo_design_predictor,
        network_factory              = network_factory,
        design_low                   = codesign["low"],
        design_high                  = codesign["high"],
        design_dim                   = len(codesign["low"]),
        num_designs                  = design_sampling["num_designs"],
        resamples_per_epoch          = design_sampling["resamples_per_epoch"],
        num_tradeoffs                = tradeoff_sampling["num_tradeoffs"],
        alpha                        = tradeoff_sampling["alpha"],
        sampling                     = tradeoff_sampling["sampling"],
        **optional_params,
        **ppo,
    )
    return train_fn, network_factory
