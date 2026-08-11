"""Shared helpers for ``codesign`` (design construction, model batching, plotting)."""

from codesign.utils.grid import (
    DesignTradeoffSampleGrid,
    DesignTradeoffRolloutGrid,
    DesignPredictorSampleGrid,
    sample_tradeoffs
)
from codesign.utils.model import (
    uniform_design_sweep,
    stack_models,
    build_batched_model,
    sample_designs,
    maximin_designs,
    min_design_gap,
    normalize_design,
    unnormalize_design,
    total_mass,
)
from codesign.utils.plotting import (
    design_colors,
    objective_labels,
    plot_design_paretos,
    plot_design_objective_pareto,
    plot_sequential_design_paretos,
    plot_mo_design_progress,
    plot_mean_hv_progress,
    MODesignTrainingPlottingInfo,
)

__all__ = [
    "DesignTradeoffSampleGrid",
    "DesignTradeoffRolloutGrid",
    "DesignPredictorSampleGrid",
    "uniform_design_sweep",
    "stack_models",
    "build_batched_model",
    "sample_designs",
    "sample_tradeoffs",
    "maximin_designs",
    "min_design_gap",
    "normalize_design",
    "unnormalize_design",
    "total_mass",
    "design_colors",
    "objective_labels",
    "plot_design_paretos",
    "plot_design_objective_pareto",
    "plot_sequential_design_paretos",
    "plot_mo_design_progress",
    "MODesignTrainingPlottingInfo",
    "plot_mean_hv_progress"
]
