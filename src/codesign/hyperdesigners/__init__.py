"""``hyperdesigners`` — design-conditioned hypernetwork RL.

``design_hypernetwork`` trains a hypernetwork that maps a robot *design* to a policy
MLP's weights (and a separate hypernetwork for the value MLP), optimizing a single
reward with brax's clipped PPO loss on a model-as-input environment (``MAICheetah``).
"""

from codesign.hyperdesigners.design_hypernetwork import train_design_hypernetwork
from codesign.hyperdesigners.mo_design_hypernetwork import (
    train_mo_design_hypernetwork,
)
from codesign.hyperdesigners.factory import (
    setup_design_hypernetwork,
    setup_mo_design_hypernetwork,
)
from codesign.hyperdesigners.networks import (
    DesignHypernetNetworks,
    make_design_hypernet_networks,
    make_design_inference_fn,
    make_mo_design_hypernet_networks,
    make_mo_design_inference_fn,
)
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    compute_design_hypernet_loss,
    compute_mo_design_hypernet_loss,
)
from codesign.hyperdesigners.acting import DesignTransition
from codesign.hyperdesigners.mo_acting import MODesignTransition

__all__ = [
    "train_design_hypernetwork",
    "train_mo_design_hypernetwork",
    "setup_design_hypernetwork",
    "setup_mo_design_hypernetwork",
    "DesignHypernetNetworks",
    "make_design_hypernet_networks",
    "make_design_inference_fn",
    "make_mo_design_hypernet_networks",
    "make_mo_design_inference_fn",
    "DesignHypernetParams",
    "compute_design_hypernet_loss",
    "compute_mo_design_hypernet_loss",
    "DesignTransition",
    "MODesignTransition",
]
