"""Evaluation utilities: single-env video rollouts and batched parallel rollouts.

``rollout_so_parallel`` scans a batched single-objective policy over a stacked, per-env
``mjx.Model`` (one model-as-input variant per env); ``rollout_design_hypernetwork`` wraps
it for a trained design-conditioned policy. For the generic single-env video driver, use
:func:`minimal_mjx.eval.rollout_policy`. Batched open-loop / adapted policies live in
:mod:`codesign.eval.policies`.
"""
from . import parallel_eval
from .parallel_eval import (
    rollout_so_parallel,
    rollout_design_hypernetwork,
    rollout_design_hypernetwork_grid,
    rollout_mo_design_hypernetwork,
    rollout_morlax,
    rollout_grid,
    TRAJECTORY_FIELDS,
)
from . import rollout_video
from .rollout_video import (
    rollout_single_video,
    rollout_design_hypernetwork_video,
    rollout_mo_design_hypernetwork_video,
    rollout_mo_design_predictor_hypernetwork_video,
    RolloutVideo,
    rollout_policy_video,
    rollout_caption,
    write_rollout_video,
    save_policy_rollout_video,
    default_video_design,
    extreme_tradeoffs_with_labels,
)

__all__ = [
    "rollout_so_parallel",
    "rollout_design_hypernetwork",
    "rollout_design_hypernetwork_grid",
    "rollout_mo_design_hypernetwork",
    "rollout_morlax",
    "rollout_grid",
    "TRAJECTORY_FIELDS",
    "rollout_single_video",
    "rollout_design_hypernetwork_video",
    "rollout_mo_design_hypernetwork_video",
    "rollout_mo_design_predictor_hypernetwork_video",
    "RolloutVideo",
    "rollout_policy_video",
    "rollout_caption",
    "write_rollout_video",
    "save_policy_rollout_video",
    "default_video_design",
    "extreme_tradeoffs_with_labels",
]
