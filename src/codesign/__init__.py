"""``codesign`` — design-conditioned (multi-objective) hypernetwork RL for robots.
"""

from . import config, envs, eval, hyperdesigners, learning, utils

# envs
from .envs import (
    CodesignBase,
    MOCodesignBase,
    CodesignMO2SO,
    RHex,
    Codesign2SingleDesign,
    MOCodesignCheetah,
    MOCodesignCheetah1D,
    MOCodesignCheetahBackLegs,
    MOCodesignCheetahFrontLegs,
    cheetah,
    load_env
)

# eval
from .eval import (
    rollout_so_parallel,
    rollout_design_hypernetwork,
    rollout_mo_design_hypernetwork,
    rollout_single_video,
    rollout_design_hypernetwork_video,
    rollout_mo_design_hypernetwork_video,
    save_policy_rollout_video,
    default_video_design,
    extreme_tradeoffs_with_labels,
)

# hyperdesigners
from .hyperdesigners import (
    train_design_hypernetwork,
    train_mo_design_hypernetwork,
    setup_design_hypernetwork,
    setup_mo_design_hypernetwork,
    DesignHypernetNetworks,
    make_design_hypernet_networks,
    make_design_inference_fn,
    make_mo_design_hypernet_networks,
    make_mo_design_inference_fn,
    DesignHypernetParams,
    compute_design_hypernet_loss,
    compute_mo_design_hypernet_loss,
    DesignTransition,
    MODesignTransition,
)

# learning
from .learning import (
    load_design_hypernetwork,
    load_mo_design_hypernetwork,
)

# utils
from .utils import (
    DesignTradeoffSampleGrid,
    DesignTradeoffRolloutGrid,
    uniform_design_sweep,
    stack_models,
    build_batched_model,
    sample_designs,
    maximin_designs,
    min_design_gap,
    normalize_design,
    total_mass,
    design_colors,
    objective_labels,
    plot_design_paretos,
    plot_design_objective_pareto,
    plot_sequential_design_paretos,
    plot_mo_design_progress,
    MODesignTrainingPlottingInfo,
    plot_cum_hv_progress
)

from .optimizers import (
    TurboState,
    TurboOptimizer
)

__all__ = [
    # subpackages
    "config",
    "envs",
    "eval",
    "hyperdesigners",
    "learning",
    "utils",
    # envs
    "CodesignBase",
    "MOCodesignBase",
    "CodesignMO2SO",
    "cheetah",
    "RHex",
    "Codesign2SingleDesign",
    "MOCodesignCheetah",
    "MOCodesignCheetah1D",
    "MOCodesignCheetahBackLegs",
    "MOCodesignCheetahFrontLegs",
    "load_env",
    # eval
    "make_open_loop_policy",
    "from_inference_fn",
    "rollout_so_parallel",
    "rollout_design_hypernetwork",
    "rollout_mo_design_hypernetwork",
    "rollout_single_video",
    "rollout_design_hypernetwork_video",
    "rollout_mo_design_hypernetwork_video",
    "save_policy_rollout_video",
    "default_video_design",
    "extreme_tradeoffs_with_labels",
    # hyperdesigners
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
    # learning
    "load_design_hypernetwork",
    "load_mo_design_hypernetwork",
    # utils
    "DesignTradeoffSampleGrid",
    "DesignTradeoffRolloutGrid",
    "uniform_design_sweep",
    "stack_models",
    "build_batched_model",
    "sample_designs",
    "maximin_designs",
    "min_design_gap",
    "normalize_design",
    "total_mass",
    "design_colors",
    "objective_labels",
    "plot_design_paretos",
    "plot_design_objective_pareto",
    "plot_sequential_design_paretos",
    "plot_mo_design_progress",
    "MODesignTrainingPlottingInfo",
    "plot_cum_hv_progress",
    # optimizers
    "TurboState",
]
