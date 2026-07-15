"""Evaluation utilities: single-env video rollouts and batched parallel rollouts.

``rollout_policy`` drives one env (video/plots). ``rollout_parallel`` scans a batched
policy over a stacked, per-env ``mjx.Model`` (one model-as-input variant per env);
``rollout_design_hypernetwork`` wraps it for a trained design-conditioned policy. Batched
open-loop / adapted policies live in :mod:`codesign.eval.policies`.
"""
from . import rollout
from .rollout import rollout_policy
from . import policies
from .policies import make_open_loop_policy, from_inference_fn
from . import parallel_eval
from . import single_eval
