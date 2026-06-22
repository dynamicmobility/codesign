"""Parallelized rollouts of a model-specific policy across a sweep of MAI variants.

Env-agnostic: works with any model-as-input (MAI) env that exposes ``generate_model(d)``
and the ``reset(rng, model)`` / ``step(state, action, model)`` signature. The generic
core :func:`rollout_parallel` stacks one ``mjx.Model`` per variant and scans every variant
forward together under a batched ``policy(obs, key, t)`` (see :mod:`codesign.eval.policies`),
recording a configurable per-step value. :func:`rollout_design_hypernetwork` is a thin
wrapper that loads a trained design-conditioned policy and records scalar reward; the
multi-objective reward is collapsed to a scalar by wrapping the env in
:class:`~codesign.envs.MAIBase.MAIMO2SO`, so the loop itself never sees the objective
dimension — that's where scalarization lives.
"""

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from codesign.envs.MAIBase import MAIMO2SO, MAIBase
from codesign.eval import policies as policy_lib
from codesign.hyperdesigners import acting
from codesign.learning.inference import load_design_hypernetwork
from codesign.utils import model as model_lib
from codesign.utils.model import uniform_design_sweep


def rollout_parallel(
    env,
    batched_model,
    policy,
    num_envs: int,
    n_steps: int,
    *,
    record_fn=None,
    mask_after_done: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Scan ``n_steps`` of a batched policy over a stacked, per-env ``mjx.Model``.

    Every env (one per stacked model) is reset, then stepped together for ``n_steps``
    under ``policy``. A configurable ``record_fn`` extracts the per-step value to log.

    Args:
        env: a model-as-input env (e.g. ``MAICheetah``), already wrapped if a scalar
            reward is desired (see :class:`~codesign.envs.MAIBase.MAIMO2SO`).
        batched_model: stacked ``mjx.Model`` with leading axis ``num_envs`` (see
            :func:`codesign.utils.model.build_batched_model`).
        policy: ``policy(obs, key, t) -> (action, extras)``, batched over the env axis
            (see :mod:`codesign.eval.policies`).
        num_envs: number of stacked models / parallel envs.
        n_steps: rollout length in env steps.
        record_fn: ``record_fn(prev_state, action, next_state) -> value`` logged each
            step; defaults to the next state's (scalar) reward.
        mask_after_done: zero the recorded value after a given env terminates. Assumes the
            value is per-env scalar (or broadcastable against ``(num_envs,)``).
        seed: base PRNG seed for resets and the action key stream.

    Returns:
        ``np.ndarray`` of stacked per-step values, shape ``(n_steps, num_envs, ...)``.
    """
    if record_fn is None:
        record_fn = lambda prev, act, nst: nst.reward

    @jax.jit
    def rollout(rngs, key):
        state = acting.reset(env, rngs, batched_model)

        def body(carry, _):
            st, k, alive, t = carry
            k, sub = jax.random.split(k)
            act, _ = policy(st.obs, sub, t)
            nst = jax.vmap(env.step, in_axes=(0, 0, 0))(st, act, batched_model)
            value = record_fn(st, act, nst)
            if mask_after_done:
                value = value * alive  # zero after termination
            alive = alive * (1.0 - nst.done)
            return (nst, k, alive, t + env.dt), value

        init = (state, key, jnp.ones(num_envs), 0.0)
        _, values = jax.lax.scan(body, init, (), length=n_steps)
        return values  # (n_steps, num_envs, ...)

    rng_reset = jax.random.split(jax.random.PRNGKey(seed + 1), num_envs)
    return np.asarray(rollout(rng_reset, jax.random.PRNGKey(seed + 2)))


def rollout_design_hypernetwork(
    env: MAIBase,
    config,
    num_envs: int,
    n_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    weighting=None,
):
    """Rollout design hypernetwork across a uniform design sweep, in parallel.
    Note that the design hypernetwork is for single-objective rewards, so this
    function expects a multi-objective env and scalarizes the reward.

    Args:
        env: a model-as-input env (e.g. ``MAICheetah``) — see module docstring.
        config: the run config dict (as written to ``config.yaml`` at train time).
        num_envs: number of designs in the uniform sweep (one env per design).
        n_steps: rollout length in env steps.
        checkpoint_path: explicit checkpoint dir; defaults to latest under ``save_dir/name``.
        seed: base PRNG seed for resets/sampling.
        deterministic: take the policy mode (vs. sampling) at each step.
        weighting: per-objective scalarization weights; defaults to the config's
            ``reward_objective_weights`` (or all-ones).

    Returns:
        ``(designs, rewards)`` where ``designs`` is ``(num_envs, design_dim)`` and
        ``rewards`` is ``(n_steps, num_envs)`` scalar per-step reward.
    """
    design = config["learning_params"]["design_params"]
    design_low = float(design["design_low"])
    design_high = float(design["design_high"])

    designs = uniform_design_sweep(config, num_envs)

    # One stacked, batched mjx.Model per design (host-side, via the env's generator).
    def generate_model_fn(design_row):
        d = float(np.asarray(design_row).reshape(-1)[0])
        return mjx.put_model(env.generate_model(d))

    batched_model = model_lib.build_batched_model(generate_model_fn, designs)
    designs_input = model_lib.normalize_design(
        jnp.asarray(designs), design_low, design_high
    )

    # Collapse the multi-objective reward to a scalar by wrapping the env.
    if weighting is None:
        weighting = config["learning_params"].get("reward_objective_weights")
    so_env = MAIMO2SO(env, weighting)

    # Load the trained, design-conditioned policy (obs/action sizes come from the
    # checkpoint's saved config) and adapt it to the (obs, key, t) rollout protocol.
    inference_fn, params = load_design_hypernetwork(config, path=checkpoint_path)
    base_policy = inference_fn(params, designs_input, deterministic=deterministic)
    policy = policy_lib.from_inference_fn(base_policy)

    rewards = rollout_parallel(
        so_env, batched_model, policy, num_envs, n_steps,
        mask_after_done=True, seed=seed,
    )
    return designs, rewards
