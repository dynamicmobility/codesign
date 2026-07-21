"""Acting utilities for ``design_hypernetwork`` and ``mo_design_hypernetwork``
(model-as-input). Adapted from ``moplayground.moppo.acting``.
"""

from typing import Any, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.training.acme.types import NestedArray
from brax.training.types import PRNGKey


class DesignTransition(NamedTuple):
    """A (single-objective) transition carrying the per-env robot design 
    alongside the usual fields."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    design: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()


def _where_done(done: jax.Array, x, y):
    """Tree-wise ``where(done, x, y)`` broadcasting ``done`` (shape ``[num_envs]``)."""

    def w(a, b):
        d = done.reshape(done.shape + (1,) * (a.ndim - done.ndim))
        return jnp.where(d, a, b)

    return jax.tree_util.tree_map(w, x, y)


def reset(env, rngs: jax.Array, models) -> Any:
    """Vmapped reset over ``(rng, model)`` plus episode bookkeeping in ``info``."""
    state = jax.vmap(env.reset, in_axes=(0, 0))(rngs, models)
    num_envs = rngs.shape[0]
    state.info["steps"] = jnp.zeros(num_envs)
    state.info["truncation"] = jnp.zeros(num_envs)
    return state


def actor_step(
    env,
    state,
    models,
    policy,
    designs: jax.Array,
    key: PRNGKey,
    first_state,
    episode_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[Any, DesignTransition]:
    """Step every env once (per-env model), build a transition, then auto-reset.
    """
    actions, policy_extras = policy(state.obs, key)
    nstate = jax.vmap(env.step, in_axes=(0, 0, 0))(state, actions, models)

    termination = nstate.done.astype(bool)  # env's own done (e.g. fall)
    steps = state.info["steps"] + 1
    reached_horizon = steps >= episode_length
    truncation = jnp.logical_and(reached_horizon, jnp.logical_not(termination))
    done = jnp.logical_or(termination, reached_horizon)

    # Record transition fields from the pre-reset (terminal) state.
    state_extras = {"truncation": truncation.astype(jnp.float32)}
    state_extras.update({x: nstate.info[x] for x in extra_fields})
    transition = DesignTransition(
        observation=state.obs,
        action=actions,
        reward=nstate.reward,
        design=designs,
        discount=1.0 - termination.astype(jnp.float32),
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

    # Update bookkeeping, then auto-reset finished envs to their per-slot first_state.
    nstate.info["truncation"] = truncation.astype(jnp.float32)
    nstate.info["steps"] = steps
    nstate = _where_done(done, first_state, nstate)
    return nstate, transition


def generate_unroll(
    env,
    state,
    models,
    policy,
    designs: jax.Array,
    key: PRNGKey,
    unroll_length: int,
    first_state,
    episode_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[Any, DesignTransition]:
    """Roll out ``unroll_length`` steps; data has leading dims ``[unroll_length, num_envs]``."""

    def f(carry, unused_t):
        cur_state, cur_key = carry
        cur_key, next_key = jax.random.split(cur_key)
        nstate, transition = actor_step(
            env,
            cur_state,
            models,
            policy,
            designs,
            cur_key,
            first_state,
            episode_length,
            extra_fields=extra_fields,
        )
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(
        f, (state, key), (), length=unroll_length
    )
    return final_state, data


class MODesignTransition(NamedTuple):
    """A multi-objective transition carrying the per-env robot ``design`` and tradeoff
    ``directive`` alongside the usual fields. ``reward`` is a per-objective vector."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    design: NestedArray
    directive: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()


def mo_actor_step(
    env,
    state,
    models,
    policy,
    designs: jax.Array,
    directives: jax.Array,
    key: PRNGKey,
    first_state,
    episode_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[Any, MODesignTransition]:
    """Step every env once (per-env model), build a vector-reward transition, auto-reset."""
    actions, policy_extras = policy(state.obs, key)
    nstate = jax.vmap(env.step, in_axes=(0, 0, 0))(state, actions, models)

    termination = nstate.done.astype(bool)  # env's own done (e.g. fall)
    steps = state.info["steps"] + 1
    reached_horizon = steps >= episode_length
    truncation = jnp.logical_and(reached_horizon, jnp.logical_not(termination))
    done = jnp.logical_or(termination, reached_horizon)

    # Record transition fields from the pre-reset (terminal) state.
    state_extras = {"truncation": truncation.astype(jnp.float32)}
    state_extras.update({x: nstate.info[x] for x in extra_fields})
    transition = MODesignTransition(
        observation=state.obs,
        action=actions,
        reward=nstate.reward,  # keep the per-objective vector
        design=designs,
        directive=directives,
        discount=1.0 - termination.astype(jnp.float32),
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

    # Update bookkeeping, then auto-reset finished envs to their per-slot first_state.
    nstate.info["truncation"] = truncation.astype(jnp.float32)
    nstate.info["steps"] = steps
    nstate = _where_done(done, first_state, nstate)
    return nstate, transition


def mo_generate_unroll(
    env,
    state,
    models,
    policy,
    designs: jax.Array,
    directives: jax.Array,
    key: PRNGKey,
    unroll_length: int,
    first_state,
    episode_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[Any, MODesignTransition]:
    """Roll out ``unroll_length`` steps; data has leading dims ``[unroll_length, num_envs]``."""

    def f(carry, unused_t):
        cur_state, cur_key = carry
        cur_key, next_key = jax.random.split(cur_key)
        nstate, transition = mo_actor_step(
            env,
            cur_state,
            models,
            policy,
            designs,
            directives,
            cur_key,
            first_state,
            episode_length,
            extra_fields=extra_fields,
        )
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(
        f, (state, key), (), length=unroll_length
    )
    return final_state, data
