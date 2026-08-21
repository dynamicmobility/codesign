"""Parallelized rollouts of a model-specific policy across a sweep of Codesign Envs.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from codesign.envs.codesign_base import CodesignMO2SO, CodesignBase, MOCodesignBase
from minimal_mjx.eval import policy as policy_lib
from codesign.hyperdesigners import acting
from codesign.learning.inference import load_design_hypernetwork, load_mo_design_hypernetwork, load_mo_design_predictor_hypernetwork
from codesign.utils import model as model_lib
from codesign.utils.grid import (
    DesignTradeoffSampleGrid,
    DesignPredictorSampleGrid,
    DesignTradeoffDataset,
    sample_tradeoffs,
)

from typing import Callable


def rollout_so_parallel(
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
    """Scan ``n_steps`` of a batched single-objective policy over a stacked, per-env ``mjx.Model``.

    For the design x tradeoff grid with per-objective accumulated reward, see
    :func:`build_grid_rollout_fn`.

    Every env (one per stacked model) is reset, then stepped together for ``n_steps``
    under ``policy``. A configurable ``record_fn`` extracts the per-step value to log.

    Args:
        env: a model-as-input env (e.g. ``CodesignCheetah``), already wrapped if a scalar
            reward is desired (see :class:`~codesign.envs.CodesignBase.CodesignMO2SO`).
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

# Per-step quantities a grid rollout can record, as ``fn(prev_state, action, next_state)``.
TRAJECTORY_FIELDS: dict[str, Callable] = {
    "reward" : lambda prev, act, nst: nst.reward,     # per-objective, before masking
    "done"   : lambda prev, act, nst: nst.done,
    "obs"    : lambda prev, act, nst: prev.obs,       # the observation ``act`` was taken on
    "action" : lambda prev, act, nst: act,
    "qpos"   : lambda prev, act, nst: nst.data.qpos,
    "qvel"   : lambda prev, act, nst: nst.data.qvel,
}


def make_record_fn(keys) -> Callable:
    """Bundle :data:`TRAJECTORY_FIELDS` entries into one ``record_fn`` returning a dict."""
    fns = {k: TRAJECTORY_FIELDS[k] for k in keys}
    return lambda prev, act, nst: {k: fn(prev, act, nst) for k, fn in fns.items()}


def _named_records(records, to_grid_axes) -> dict:
    """Flatten a record pytree to ``{name: array}``, naming nested leaves by their path.

    A field can itself be a pytree -- ``obs`` is a dict for envs with several observation
    streams -- so ``{'obs': {'state': x}}`` becomes ``{'obs.state': x}`` and every entry
    of ``DesignTradeoffDataset.data`` stays a plain array.
    """
    leaves = jax.tree_util.tree_flatten_with_path(records)[0]
    return {
        ".".join(str(getattr(k, "key", k)) for k in path): to_grid_axes(value)
        for path, value in leaves
    }


def _grid_rollout(env, n_steps, make_policy, deterministic, record_fn):
    """One ``(key, design, tradeoff, model)`` episode, for the grid vmaps below.

    Returns ``rollout(key, design, tradeoff, model, params) -> ((final_state, key, ret),
    records)``, where ``ret`` is the per-objective reward accumulated over the episode
    (zeroed from the terminating step on) and ``records`` is ``record_fn``'s output
    stacked along a leading ``n_steps`` axis.
    """
    if record_fn is None:
        record_fn = lambda prev, act, nst: {}

    def step_fn(carry, _, model, policy):
        state, key, total = carry
        key, sub = jax.random.split(key)
        action, _ = policy(state.obs, sub)
        new_state = env.step(state, action, model)
        record = record_fn(state, action, new_state)
        # accumulate per-objective reward, masking out steps after termination
        return (new_state, key, total + new_state.reward * (1 - new_state.done)), record

    def rollout(key, design, tradeoff, model, params):
        policy = make_policy(
            params         = params,
            designs        = design,
            tradeoffs      = tradeoff,
            deterministic  = deterministic,
        )
        scan_step_fn = functools.partial(step_fn, model=model, policy=policy)
        key_reset, key_act = jax.random.split(key)
        state = env.reset(key_reset, model)
        carry = (state, key_act, state.reward)
        return jax.lax.scan(scan_step_fn, carry, (), n_steps)

    return rollout


def build_grid_rollout_fn(
    env, n_steps, make_policy, deterministic=True, record_fn=None
):
    """Build a jitted rollout of a design/tradeoff-conditioned hypernetwork policy,
    vmapped over grid axes (design, tradeoff, rep).

    The returned function has signature ``rollout(keys, designs, tradeoffs, models,
    params)`` with ``keys`` of shape ``(n_designs, n_tradeoffs, per_cell, 2)``,
    ``designs`` ``(n_designs, design_dim)``, ``tradeoffs`` ``(n_tradeoffs, num_objectives)``
    and ``models`` batched per design. Outputs keep the leading
    ``(n_designs, n_tradeoffs, per_cell)`` grid axes.
    """
    rollout = _grid_rollout(env, n_steps, make_policy, deterministic, record_fn)
    over_reps      = jax.vmap(rollout,    in_axes=(0, None, None, None, None))
    over_tradeoffs = jax.vmap(over_reps,  in_axes=(0, None, 0, None, None))
    over_designs   = jax.vmap(over_tradeoffs, in_axes=(0, 0, None, 0, None))
    return jax.jit(over_designs)


def build_predictor_rollout_fn(
    env, n_steps, make_policy, deterministic=True, record_fn=None
):
    """As :func:`build_grid_rollout_fn`, but with designs *paired* with tradeoffs instead
    of crossed with them: ``designs[t, g]`` was drawn for ``tradeoffs[t]`` only.

    Signature ``rollout(keys, designs, tradeoffs, models, params)`` with ``keys`` of shape
    ``(n_tradeoffs, group_size, per_cell, 2)``, ``designs`` ``(n_tradeoffs, group_size,
    design_dim)``, ``tradeoffs`` ``(n_tradeoffs, num_objectives)`` and ``models`` batched
    over ``(n_tradeoffs, group_size)``. Outputs keep those three leading axes.
    """
    rollout = _grid_rollout(env, n_steps, make_policy, deterministic, record_fn)
    over_reps      = jax.vmap(rollout,   in_axes=(0, None, None, None, None))
    over_group     = jax.vmap(over_reps, in_axes=(0, 0, None, 0, None))
    over_tradeoffs = jax.vmap(over_group, in_axes=(0, 0, 0, 0, None))
    return jax.jit(over_tradeoffs)


def rollout_mo_design_hypernetwork(
    env: CodesignBase,
    config,
    n_designs: int,
    n_tradeoffs: int,
    per_cell: int,
    n_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    design_predictor: bool = False,
    record=(),
    sampling: str = "sparse-heavytail",
) -> DesignTradeoffDataset:
    # TODO: clean up this function and make it human-readable
    """Roll out a trained MO design hypernetwork over a design x tradeoff grid.

    Tradeoffs are :func:`sample_tradeoffs` draws on the objective simplex. Designs are
    either a space-filling sweep of the configured design range (``design_predictor=False``)
    or drawn from the trained design predictor ``f(d | w)`` for each tradeoff
    (``design_predictor=True``); in the latter case the ``n_designs`` designs of a tradeoff
    belong to that tradeoff alone and are *not* rolled out against the others. Each grid
    cell is rolled out ``per_cell`` times for ``n_steps`` env steps.

    Args:
        env: a model-as-input env (e.g. ``CodesignCheetah``).
        config: the run config dict (as written to ``config.yaml`` at train time).
        n_designs: designs per tradeoff (the whole sweep when ``design_predictor=False``).
        n_tradeoffs: number of sampled tradeoffs.
        per_cell: rollout repetitions per (design, tradeoff) cell. Only informative when
            the env's reset or the policy is stochastic; the cheetah reset is neither.
        n_steps: rollout length in env steps.
        checkpoint_path: explicit checkpoint dir; defaults to latest under ``save_dir/name``.
        seed: base PRNG seed for resets, tradeoff sampling and design sampling.
        deterministic: take the policy mode (vs. sampling) at each step.
        design_predictor: draw designs from ``f(d | w)`` instead of sweeping the design box.
            With ``n_designs == 1`` the predictor's mode (its predicted optimum) is used;
            with more, designs are sampled from ``f`` so the group spreads around it.
        record: :data:`TRAJECTORY_FIELDS` keys to keep per step, e.g. ``("reward", "obs")``.
        sampling: tradeoff sampling style; see :func:`sample_tradeoffs`.

    Returns:
        A :class:`DesignTradeoffDataset` whose ``rewards`` are the accumulated
        per-objective returns, shape ``(n_designs, n_tradeoffs, per_cell, num_objectives)``,
        and whose ``data[key]`` holds each recorded field with a leading
        ``(n_designs, n_tradeoffs, per_cell, n_steps)``.
    """
    # Which checkpoint the run wrote is set by its algorithm, not by `design_predictor`:
    # a predictor run's saved network config carries the extra predictor kwargs, so its
    # policy hypernetwork can only be rebuilt by the predictor factory.
    if config["algorithm"] == "mo_design_predictor_hypernetwork":
        make_policy_fn, design_predictor_inference_fn, params = (
            load_mo_design_predictor_hypernetwork(config, path=checkpoint_path)
        )
    elif design_predictor:
        raise ValueError(
            f"design_predictor=True needs a 'mo_design_predictor_hypernetwork' run; "
            f"this one is '{config['algorithm']}'."
        )
    else:
        make_policy_fn, params = load_mo_design_hypernetwork(
            config, path=checkpoint_path
        )

    if design_predictor:
        tradeoffs = sample_tradeoffs(
            np.random.default_rng(seed), 0, n_tradeoffs, len(env.objectives),
            sampling=sampling,
        )
        limits = np.asarray(env.design_limits)
        # One predictor draw per (tradeoff, group member).
        tradeoffs_tiled = jnp.repeat(jnp.asarray(tradeoffs), n_designs, axis=0)
        designs_input, _ = design_predictor_inference_fn(
            params[2],
            tradeoffs_tiled,
            deterministic = n_designs == 1,
            key_sample    = jax.random.PRNGKey(seed + 1),
        )
        designs_input = designs_input.reshape(n_tradeoffs, n_designs, -1)
        # Designs come out normalized to [0, 1]; the model generator wants physical units.
        designs = model_lib.unnormalize_design(designs_input, limits[0], limits[1])
        grid = DesignPredictorSampleGrid(
            designs=np.asarray(designs), tradeoffs=tradeoffs, per_cell=per_cell
        )
        build_rollout_fn = build_predictor_rollout_fn
        grid_shape = (grid.n_tradeoffs, grid.group_size, grid.per_cell)
        # (tradeoff, group, rep) -> the (design, tradeoff, rep) convention of the dataset
        to_grid_axes = lambda x: np.swapaxes(np.asarray(x), 0, 1)
    else:
        grid = DesignTradeoffSampleGrid.from_uniform_sample(
            env, seed=seed, n_tradeoffs=n_tradeoffs, n_designs=n_designs,
            per_cell=per_cell, sampling=sampling,
        )
        tradeoffs = grid.tradeoffs
        designs_input = model_lib.normalize_design(
            jnp.asarray(grid.designs), config=config
        )
        build_rollout_fn = build_grid_rollout_fn
        grid_shape = (grid.n_designs, grid.n_tradeoffs, grid.per_cell)
        to_grid_axes = np.asarray

    batched_model = grid.build_models(env, tiled=False)
    keys = jax.random.split(
        jax.random.PRNGKey(seed + 2), grid.num_envs
    ).reshape(*grid_shape, -1)

    rollout_fn = build_rollout_fn(
        env           = env,
        n_steps       = n_steps,
        make_policy   = make_policy_fn,
        deterministic = deterministic,
        record_fn     = make_record_fn(record),
    )
    (_, _, final_rewards), records = rollout_fn(
        keys, designs_input, jnp.asarray(tradeoffs), batched_model, params
    )

    return DesignTradeoffDataset(
        designs       = to_grid_axes(grid.designs),
        tradeoffs     = np.asarray(tradeoffs),
        rewards       = to_grid_axes(final_rewards),
        objectives    = env.objectives,
        data          = _named_records(records, to_grid_axes),
    )


def rollout_design_hypernetwork(
    env: CodesignBase | MOCodesignBase,
    config,
    num_envs: int,
    num_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    weighting=None,
    trials_per_env: int = 1
):
    """Rollout design hypernetwork across a uniform design sweep, in parallel.
    Note that the design hypernetwork is for single-objective rewards, so this
    function will scalarize the reward of a MOCodesignBase policy.

    Args:
        env: a model-as-input env (e.g. ``CodesignCheetah``) — see module docstring.
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
        ``rewards`` is ``(trials_per_env, n_steps, num_envs)`` scalar per-step reward.
    """
    if trials_per_env < 1:
        raise ValueError("trials_per_env must be at least 1")
    
    # Collapse the multi-objective reward to a scalar by wrapping the env.
    if weighting is None:
        weighting = config["learning_params"].get("reward_objective_weights")
    
    if type(env) == MOCodesignBase:
        env = CodesignMO2SO(env, weighting)

    codesign = config["env_config"]["codesign"]
    design_low = np.asarray(codesign["low"])
    design_high = np.asarray(codesign["high"])
    design_dim = len(codesign["low"])

    designs = model_lib.sample_designs(
        rng         = np.random.default_rng(seed),
        num_envs    = num_envs,
        low         = design_low,
        high        = design_high,
        dim         = design_dim,
    )
    repeated_designs = np.repeat(designs, trials_per_env, axis=0)

    # One stacked, batched mjx.Model per design (host-side, via the env's generator).
    batched_model = model_lib.build_batched_model(env, repeated_designs)
    designs_input = model_lib.normalize_design(
        jnp.asarray(repeated_designs), design_low, design_high
    )

    # Load the trained, design-conditioned policy (obs/action sizes come from the
    # checkpoint's saved config) and adapt it to the (obs, key, t) rollout protocol.
    inference_fn, params = load_design_hypernetwork(config, path=checkpoint_path)
    base_policy = inference_fn(params, designs_input, deterministic=deterministic)
    policy = policy_lib.from_inference_fn(base_policy)

    rewards = rollout_so_parallel(
        env,
        batched_model,
        policy,
        num_envs * trials_per_env,
        num_steps,
        mask_after_done=True,
        seed=seed,
    )
    rewards = rewards.reshape(num_steps, num_envs, trials_per_env).transpose(2, 0, 1)
    return designs, rewards
