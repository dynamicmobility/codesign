"""Shared helpers for ``codesign`` (design construction, model batching, plotting)."""

from codesign.utils.grid import (
    DesignTradeoffSampleGrid,
    DesignTradeoffRolloutGrid,
)
from codesign.utils.model import (
    uniform_design_sweep,
    stack_models,
    build_batched_model,
    sample_designs,
    normalize_design,
    total_mass,
)
from codesign.utils.plotting import (
    design_colors,
    objective_labels,
    plot_design_paretos,
    plot_design_objective_pareto,
    plot_sequential_design_paretos,
    plot_mo_design_progress,
    plot_cum_hv_progress,
    MODesignTrainingPlottingInfo,
)

__all__ = [
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
    "plot_cum_hv_progress"
]
