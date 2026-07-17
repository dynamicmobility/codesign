"""Evaluation utilities: single-env video rollouts and batched parallel rollouts.

``rollout_policy`` drives one env (video/plots). ``rollout_parallel`` scans a batched
policy over a stacked, per-env ``mjx.Model`` (one model-as-input variant per env);
``rollout_design_hypernetwork`` wraps it for a trained design-conditioned policy. Batched
open-loop / adapted policies live in :mod:`codesign.eval.policies`.
"""
from . import rollout
from .rollout import rollout_policy
from . import parallel_eval
from .parallel_eval import (
    rollout_parallel,
    rollout_design_hypernetwork,
    rollout_mo_designs,
)
from . import single_eval
from .single_eval import (
    rollout_single,
    rollout_design_hypernetwork,
    rollout_mo_design_hypernetwork,
)

# NOTE: ``rollout_design_hypernetwork`` is defined in both ``parallel_eval`` and
# ``single_eval``; the ``single_eval`` import above intentionally wins here.
__all__ = [
    "rollout_policy",
    "rollout_parallel",
    "rollout_mo_designs",
    "rollout_single",
    "rollout_design_hypernetwork",
    "rollout_mo_design_hypernetwork",
]
