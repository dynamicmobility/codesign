"""``hyperdesigners`` — design-conditioned RL.

``variants`` holds one module per algorithm, each with that algorithm's own networks,
loss, training loop and config wiring; ``networks``/``losses`` keep only the pieces
several algos share, and ``shared``/``acting`` the PPO scaffolding and rollout loop they
all run on.

The algos form a chain, each reusing the one before it: ``design_hypernetwork`` maps a
design to a policy/value MLP's weights, ``mo_design_hypernetwork`` widens that to
``[design, tradeoff]``, and ``mo_design_predictor_hypernetwork`` adds a predictor that
proposes the designs. ``design_mlp`` stands apart: it conditions on the design by
appending it to the observation instead of generating weights.
"""

from codesign.hyperdesigners.variants.design_mlp import (
    DesignMLPParams,
    DesignNetworks,
    compute_design_mlp_loss,
    make_design_mlp_inference_fn,
    make_design_mlp_networks,
    setup_design_mlp,
    train_design_mlp,
)
from codesign.hyperdesigners.variants.design_hypernetwork import (
    compute_design_hypernet_loss,
    make_design_hypernet_networks,
    load_warmup_params,
    make_design_hypernetwork,
    make_design_inference_fn,
    setup_design_hypernetwork,
    train_design_hypernetwork,
)
from codesign.hyperdesigners.variants.design_lookup_hypernetwork import (
    make_lookup_hypernet_networks,
    make_lookup_hypernetwork,
    warn_off_table,
    setup_design_lookup_hypernetwork,
    train_design_lookup_hypernetwork,
)
from codesign.hyperdesigners.variants.mo_design_hypernetwork import (
    compute_mo_design_hypernet_loss,
    make_mo_design_hypernet_networks,
    make_mo_design_inference_fn,
    setup_mo_design_hypernetwork,
    train_mo_design_hypernetwork,
)
from codesign.hyperdesigners.variants.mo_design_predictor_hypernetwork import (
    DesignPredictorHypernetNetworks,
    DesignPredictorTransition,
    compute_grpo_loss,
    make_design_predictor_inference_fn,
    make_mo_design_predictor_hypernet_networks,
    setup_mo_design_predictor_hypernetwork,
    train_mo_design_predictor,
)
from codesign.hyperdesigners.hypernetworks import (
    DualA2CHypernet,
    LookupA2CHypernet,
    sobol_design_table,
)
from codesign.hyperdesigners.networks import (
    DesignHypernetNetworks,
    FeedForwardHypernetwork,
    make_value_fn,
    make_vector_value_network,
)
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    huber_loss,
    mse_loss,
)
from codesign.hyperdesigners.shared import (
    paired_eval_keys,
    per_design_metrics,
)
from codesign.hyperdesigners.acting import DesignTransition

__all__ = [
    # design_mlp
    "train_design_mlp",
    "setup_design_mlp",
    "make_design_mlp_networks",
    "make_design_mlp_inference_fn",
    "DesignNetworks",
    "DesignMLPParams",
    "compute_design_mlp_loss",
    # design_hypernetwork
    "train_design_hypernetwork",
    "setup_design_hypernetwork",
    "make_design_hypernetwork",
    "make_design_hypernet_networks",
    "make_design_inference_fn",
    "compute_design_hypernet_loss",
    "load_warmup_params",
    # design_lookup_hypernetwork
    "train_design_lookup_hypernetwork",
    "setup_design_lookup_hypernetwork",
    "make_lookup_hypernet_networks",
    "make_lookup_hypernetwork",
    "warn_off_table",
    # mo_design_hypernetwork
    "train_mo_design_hypernetwork",
    "setup_mo_design_hypernetwork",
    "make_mo_design_hypernet_networks",
    "make_mo_design_inference_fn",
    "compute_mo_design_hypernet_loss",
    # mo_design_predictor_hypernetwork
    "train_mo_design_predictor",
    "setup_mo_design_predictor_hypernetwork",
    "make_mo_design_predictor_hypernet_networks",
    "make_design_predictor_inference_fn",
    "DesignPredictorHypernetNetworks",
    "DesignPredictorTransition",
    "compute_grpo_loss",
    # shared
    "DualA2CHypernet",
    "LookupA2CHypernet",
    "sobol_design_table",
    "DesignHypernetNetworks",
    "FeedForwardHypernetwork",
    "make_value_fn",
    "make_vector_value_network",
    "DesignHypernetParams",
    "mse_loss",
    "huber_loss",
    "paired_eval_keys",
    "per_design_metrics",
    "DesignTransition",
]
