"""``codesign`` — design-conditioned (multi-objective) hypernetwork RL for robots.

Subpackages remain importable as ``codesign.envs``, ``codesign.eval``,
``codesign.hyperdesigners``, ``codesign.learning``, ``codesign.utils`` and
``codesign.config``. For convenience each subpackage's public API is also re-exported at
the top level, so e.g. ``codesign.train_mo_design_hypernetwork`` and
``codesign.plot_design_paretos`` work directly. ``codesign.config`` is the exception:
its names stay namespaced (``codesign.config.Train``, not ``codesign.Train``) because
they are generic enough to collide with the rest of the API.

Re-exports are written out explicitly (rather than looped) so static tooling
(Pylance/Pyright, IDE autocomplete) can see them. Keep this file in sync with the
subpackage ``__all__`` lists; a name exported by two subpackages would show up here as a
shadowed re-import.
"""

from . import config, envs, eval, hyperdesigners, learning, utils

# envs
from .envs import (
    CodesignBase,
    CodesignMO2SO,
    CodesignCheetah,
    RHex,
    Codesign2SingleDesign,
    MOCodesignCheetah
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
    "CodesignMO2SO",
    "CodesignCheetah",
    "RHex",
    "Codesign2SingleDesign",
    "MOCodesignCheetah",
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
]
