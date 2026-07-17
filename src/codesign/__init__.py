"""``codesign`` — design-conditioned (multi-objective) hypernetwork RL for robots.

Subpackages remain importable as ``codesign.envs``, ``codesign.eval``,
``codesign.hyperdesigners``, ``codesign.learning`` and ``codesign.utils``. For
convenience each subpackage's public API is also re-exported at the top level, so e.g.
``codesign.train_mo_design_hypernetwork`` and ``codesign.plot_design_paretos`` work
directly.

Re-exports are written out explicitly (rather than looped) so static tooling
(Pylance/Pyright, IDE autocomplete) can see them. Keep this file in sync with the
subpackage ``__all__`` lists; a name exported by two subpackages would show up here as a
shadowed re-import.
"""

from . import envs, eval, hyperdesigners, learning, utils

# envs
from .envs import (
    CodesignBase,
    CodesignMO2SO,
    CodesignCheetah,
    RHex,
)

# eval
from .eval import (
    rollout_policy,
    make_open_loop_policy,
    from_inference_fn,
    rollout_parallel,
    rollout_mo_designs,
    rollout_single,
    rollout_design_hypernetwork,
    rollout_mo_design_hypernetwork,
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
    normalize_design,
    total_mass,
    design_colors,
    objective_labels,
    plot_design_paretos,
    plot_design_objective_pareto,
    plot_sequential_design_paretos,
    plot_mo_design_progress,
    MODesignTrainingPlottingInfo,
)

__all__ = [
    # subpackages
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
    # eval
    "rollout_policy",
    "make_open_loop_policy",
    "from_inference_fn",
    "rollout_parallel",
    "rollout_mo_designs",
    "rollout_single",
    "rollout_design_hypernetwork",
    "rollout_mo_design_hypernetwork",
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
    "normalize_design",
    "total_mass",
    "design_colors",
    "objective_labels",
    "plot_design_paretos",
    "plot_design_objective_pareto",
    "plot_sequential_design_paretos",
    "plot_mo_design_progress",
    "MODesignTrainingPlottingInfo",
]
