"""Parallelized rollouts of a model-specific policy across a sweep of Codesign Envs.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from moplayground.learning.inference import load_hypernetwork_inference_fn

from codesign.envs.codesign_base import CodesignMO2SO, CodesignBase, MOCodesignBase
from minimal_mjx.eval import policy as policy_lib
from codesign.hyperdesigners import acting
from codesign.learning.inference import load_design_hypernetwork, load_mo_design_hypernetwork, load_mo_design_predictor_hypernetwork, load_mo_design_value_hypernetwork
from codesign.utils import model as model_lib
from codesign.utils.grid import Grid

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


def _named_records(records) -> dict:
    """Flatten a record pytree to ``{name: array}``, naming nested leaves by their path.

    A field can itself be a pytree -- ``obs`` is a dict for envs with several observation
    streams -- so ``{'obs': {'state': x}}`` becomes ``{'obs.state': x}`` and every entry
    of ``Grid.data`` stays a plain array.
    """
    leaves = jax.tree_util.tree_flatten_with_path(records)[0]
    return {
        ".".join(str(getattr(k, "key", k)) for k in path): np.asarray(value)
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
    vmapped over the grid's ``(design, tradeoff, repetition)`` axes.

    The returned function has signature ``rollout(keys, designs, tradeoffs, models,
    params)`` with ``keys`` of shape ``(M, K, C, 2)`` and ``designs`` ``(M, K, design_dim)``,
    ``tradeoffs`` ``(M, K, n_r)`` and ``models`` all carrying the grid's ``(M, K)`` cell
    axes. Outputs keep the leading ``(M, K, C)`` grid axes.
    """
    rollout = _grid_rollout(env, n_steps, make_policy, deterministic, record_fn)
    over_reps      = jax.vmap(rollout,        in_axes=(0, None, None, None, None))
    over_tradeoffs = jax.vmap(over_reps,      in_axes=(0, 0, 0, 0, None))
    over_designs   = jax.vmap(over_tradeoffs, in_axes=(0, 0, 0, 0, None))
    return jax.jit(over_designs)


def _load_mo_design_networks(config, checkpoint_path):
    """The run's policy hypernetwork, and its design predictor when it trained one.


    Returns ``(make_policy_fn, design_predictor_inference_fn, params)``, the middle entry
    being ``None`` for a run without a predictor.
    """
    if config["algorithm"] == "mo_design_predictor_hypernetwork":
        return load_mo_design_predictor_hypernetwork(config, path=checkpoint_path)
    make_policy_fn, params = load_mo_design_hypernetwork(config, path=checkpoint_path)
    return make_policy_fn, None, params


def make_design_policy_fn(inference_fn):
    """The grid rollout's ``make_policy`` for a design-conditioned single-objective
    hypernetwork: it reads the cell's design, the cell's tradeoff being no part of what it
    was trained on (see :func:`rollout_design_hypernetwork_grid`).
    """
    def make_policy(params, designs, tradeoffs, deterministic=True):
        return inference_fn(params, designs, deterministic=deterministic)

    return make_policy


def make_morlax_policy_fn(inference_fn):
    """The grid rollout's ``make_policy`` for morlax: its hypernetwork ``H(w)`` is keyed on
    the tradeoff alone, morlax having trained on the one design its env was built with.
    """
    def make_policy(params, designs, tradeoffs, deterministic=True):
        return inference_fn(params, tradeoffs, deterministic=deterministic)

    return make_policy


def grid_keys(grid: Grid, seed: int):
    """One PRNG key per cell repetition, shaped ``grid.cell_shape + (2,)``."""
    return jax.random.split(
        jax.random.PRNGKey(seed + 2), grid.num_envs
    ).reshape(*grid.cell_shape, -1)


def rollout_grid(
    env: CodesignBase,
    config,
    grid: Grid,
    n_steps: int,
    make_policy,
    params,
    *,
    seed: int = 0,
    deterministic: bool = True,
    record=(),
) -> Grid:
    """Roll a design/tradeoff-conditioned policy out over every cell of ``grid``, in parallel.

    How designs and tradeoffs pair up is the grid's own: a crossed grid rolls every design
    out against every tradeoff, while a predictor grid's ``designs[:, k]`` were drawn from
    ``f(d | w_k)`` and meet that tradeoff alone. Each cell is rolled out ``grid.per_cell``
    times for ``n_steps`` steps, on the model that cell's design builds.

    Args:
        env: a model-as-input env (e.g. ``CodesignCheetah``).
        config: the run config dict (as written to ``config.yaml`` at train time).
        grid: the design x tradeoff grid to roll out, in physical design units.
        n_steps: rollout length in env steps.
        make_policy: ``make_policy(params, designs, tradeoffs, deterministic) ->
            policy(obs, key)``, built per cell by :func:`build_grid_rollout_fn`.
        params: the trained parameters ``make_policy`` reads.
        seed: base PRNG seed for the rollout resets and action streams.
        deterministic: take the policy mode (vs. sampling) at each step.
        record: :data:`TRAJECTORY_FIELDS` keys to keep per step, e.g. ``("reward", "obs")``.

    Returns:
        A copy of ``grid`` with ``rewards`` filled in -- the accumulated per-objective
        returns, shape ``(M, K, C, n_r)`` -- whose recorded ``data[key]`` fields carry a
        leading ``(M, K, C, n_steps)``.
    """
    batched_model = grid.build_models(env, tiled=False)
    # The policy is conditioned on designs in [0, 1]; the grid holds physical units.
    designs_input = model_lib.normalize_design(jnp.asarray(grid.designs), config=config)

    rollout_fn = build_grid_rollout_fn(
        env           = env,
        n_steps       = n_steps,
        make_policy   = make_policy,
        deterministic = deterministic,
        record_fn     = make_record_fn(record),
    )
    (_, _, final_rewards), records = rollout_fn(
        grid_keys(grid, seed), designs_input, jnp.asarray(grid.tradeoffs),
        batched_model, params,
    )
    return dataclasses.replace(
        grid,
        rewards    = np.asarray(final_rewards),
        objectives = env.objectives,
        data       = _named_records(records),
    )


def rollout_mo_design_hypernetwork(
    env: CodesignBase,
    config,
    grid: Grid,
    n_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    record=(),
) -> Grid:
    """Roll out a trained MO design hypernetwork over a sampled design x tradeoff grid.

    Its policy is keyed on both axes of a cell, so the grid is swept as :func:`rollout_grid`
    describes. Arguments are that function's, minus the policy it loads here.

    Returns:
        The rolled-out grid, whose ``data["value"]`` additionally holds the value
        hypernetwork's initial-state prediction per cell.
    """
    make_policy_fn, _, params = _load_mo_design_networks(config, checkpoint_path)
    rolled = rollout_grid(
        env, config, grid, n_steps, make_policy_fn, params,
        seed=seed, deterministic=deterministic, record=record,
    )

    value_inference_fn, value_params = load_mo_design_value_hypernetwork(
        config, path=checkpoint_path
    )
    flat_designs, flat_tradeoffs = grid.flatten()
    reset_keys = jax.vmap(jax.random.split)(grid_keys(grid, seed).reshape(-1, 2))[:, 0]
    initial_states = jax.vmap(env.reset)(
        reset_keys, grid.build_models(env, tiled=True)
    )
    value_fn = value_inference_fn(
        value_params,
        model_lib.normalize_design(jnp.asarray(flat_designs), config=config),
        jnp.asarray(flat_tradeoffs),
    )
    values = grid.unflatten(np.asarray(value_fn(initial_states.obs)))

    return dataclasses.replace(rolled, data={**rolled.data, "value": values})


def rollout_design_hypernetwork_grid(
    env: CodesignBase,
    config,
    grid: Grid,
    n_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    record=(),
) -> Grid:
    """Roll out a trained (single-objective) design hypernetwork over a design sweep grid.

    The policy is keyed on the design alone, so the grid's single tradeoff column is never
    fed to it: it records which scalarization the run was trained under, and is what
    ``grid.scalarized_rewards`` weighs the env's per-objective returns by.

    Unlike :func:`rollout_design_hypernetwork`, which returns per-step scalar rewards over a
    design sweep, this fills in a :class:`~codesign.utils.grid.Grid`, so every hypernetwork's
    dataset shares one layout. Arguments are :func:`rollout_grid`'s, minus the policy it
    loads here.
    """
    inference_fn, params = load_design_hypernetwork(config, path=checkpoint_path)
    return rollout_grid(
        env, config, grid, n_steps, make_design_policy_fn(inference_fn), params,
        seed=seed, deterministic=deterministic, record=record,
    )


def rollout_morlax(
    env: MOCodesignBase,
    config,
    grid: Grid,
    n_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    record=(),
) -> Grid:
    """Roll out a trained morlax hypernetwork over a tradeoff sweep grid.

    morlax keys its policy on the tradeoff alone and trains against a single design, so the
    grid it sweeps is that one design (``env_config.codesign.default_design``) against ``K``
    tradeoffs -- a ``(1, K, C, n_r)`` grid. The rollout is otherwise the hypernetworks':
    arguments are :func:`rollout_grid`'s, minus the policy it loads here.
    """
    inference_fn, params = load_hypernetwork_inference_fn(
        config, path=None if checkpoint_path is None else epath.Path(checkpoint_path)
    )
    return rollout_grid(
        env, config, grid, n_steps, make_morlax_policy_fn(inference_fn), params,
        seed=seed, deterministic=deterministic, record=record,
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

    # An (M, 1, trials_per_env) grid: one design axis, the single trivial tradeoff.
    grid = Grid.from_design_sample(env, seed, num_envs, per_cell=trials_per_env)
    batched_model, designs_input, _ = grid.env_inputs(env)

    # Load the trained, design-conditioned policy (obs/action sizes come from the
    # checkpoint's saved config) and adapt it to the (obs, key, t) rollout protocol.
    inference_fn, params = load_design_hypernetwork(config, path=checkpoint_path)
    base_policy = inference_fn(params, designs_input, deterministic=deterministic)
    policy = policy_lib.from_inference_fn(base_policy)

    rewards = rollout_so_parallel(
        env,
        batched_model,
        policy,
        grid.num_envs,
        num_steps,
        mask_after_done=True,
        seed=seed,
    )
    rewards = rewards.reshape(num_steps, num_envs, trials_per_env).transpose(2, 0, 1)
    return grid.designs[:, 0], rewards
