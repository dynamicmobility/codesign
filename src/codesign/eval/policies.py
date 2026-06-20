"""Batched policies for parallel model-as-input rollouts.

Every policy here follows one protocol::

    policy(obs, key, t) -> (action, extras)

where ``obs`` is the env observation, ``key`` a PRNG key, ``t`` the current env time
(seconds), and ``action`` matches the obs's batch shape with a trailing ``action_size``.
The ``t`` argument lets open-loop, time-dependent policies (sinusoids) share the same
loop as closed-loop ones; policies that don't need it ignore it.

The batch shape is read off the obs (its leading dims, i.e. everything but the feature
axis), so the *same* policy works for a batched parallel rollout (``obs`` leaf
``(num_envs, obs_dim)`` -> action ``(num_envs, action_size)``) and a single-instance
rollout (``obs`` leaf ``(obs_dim,)`` -> action ``(action_size,)``). Callers only supply
``action_size``.

The open-loop factories mirror the ``'zero'`` / ``'random'`` naming of
``codesign.eval.rollout.make_dummy_inference_fn`` (the single-env, non-jitted analogue).
"""

import jax
import jax.numpy as jnp


def _batch_shape(obs) -> tuple:
    """Leading (batch) dims of the observation — all axes but the trailing feature axis.

    ``()`` for a single env, ``(num_envs,)`` for a batched rollout.
    """
    return jax.tree_util.tree_leaves(obs)[0].shape[:-1]


def make_zero_policy(action_size: int):
    """Open-loop policy that always emits zero actions."""

    def policy(obs, key, t):
        return jnp.zeros((*_batch_shape(obs), action_size)), {}

    return policy


def make_random_policy(action_size: int):
    """Open-loop policy that emits uniform random actions in ``[-1, 1]``."""

    def policy(obs, key, t):
        return (
            jax.random.uniform(
                key, (*_batch_shape(obs), action_size), minval=-1.0, maxval=1.0
            ),
            {},
        )

    return policy


def make_sinusoidal_policy(
    action_size: int, amp: float = 0.8, freq: float = 1.5, phases=None
):
    """Open-loop sinusoid ``amp * sin(2*pi*freq*t + phases)``, identical across envs.

    ``phases`` defaults to ``linspace(0, pi, action_size)`` so the actuators are spread
    out of phase (the same default as the original ``rollout_mai_cheetah`` sinusoid).
    """
    phases = (
        jnp.linspace(0.0, jnp.pi, action_size) if phases is None else jnp.asarray(phases)
    )

    def policy(obs, key, t):
        action = amp * jnp.sin(2.0 * jnp.pi * freq * t + phases)  # (action_size,)
        return jnp.broadcast_to(action, (*_batch_shape(obs), action_size)), {}

    return policy


def from_inference_fn(base_policy):
    """Adapt a brax ``policy(obs, key) -> (action, extras)`` to the ``(obs, key, t)`` protocol.

    Used to drop a trained (e.g. design-conditioned) policy into ``rollout_parallel``; the
    time argument is ignored.
    """

    def policy(obs, key, t):
        return base_policy(obs, key)

    return policy


_OPEN_LOOP = {
    "zero": make_zero_policy,
    "random": make_random_policy,
    "sinusoid": make_sinusoidal_policy,
}


def make_open_loop_policy(kind: str, action_size: int, **kwargs):
    """Build an open-loop policy by name (``'zero'``, ``'random'``, ``'sinusoid'``).

    Extra kwargs are forwarded to the underlying factory (e.g. ``amp`` / ``freq`` /
    ``phases`` for ``'sinusoid'``); they are ignored by factories that don't accept them.
    """
    if kind not in _OPEN_LOOP:
        raise ValueError(
            f"Unknown open-loop policy '{kind}'. Expected one of {list(_OPEN_LOOP)}."
        )
    factory = _OPEN_LOOP[kind]
    if kind == "sinusoid":
        return factory(action_size, **kwargs)
    return factory(action_size)
