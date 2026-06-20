"""Parallelized rollout of a trained design hypernetwork across a sweep of designs.

Env-agnostic: works with any model-as-input (MAI) env that exposes ``generate_model(d)``
and the ``reset(rng, model)`` / ``step(state, action, model)`` signature. The
multi-objective reward is collapsed to a scalar by wrapping the env in
:class:`~codesign.envs.MAIBase.MAIMO2SO`, so the rollout loop itself never sees the
objective dimension — that's where scalarization lives.
"""

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from codesign.envs.MAIBase import MAIMO2SO, MAIBase
from codesign.hyperdesigners import acting
from codesign.learning.inference import load_design_hypernetwork
from codesign.utils import model as model_lib
from codesign.utils.model import uniform_design_sweep


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

    # Load the trained policy (obs/action sizes come from the checkpoint's saved config).
    inference_fn, params = load_design_hypernetwork(config, path=checkpoint_path)

    @jax.jit
    def rollout(rngs, key):
        state = acting.reset(so_env, rngs, batched_model)
        policy = inference_fn(params, designs_input, deterministic=deterministic)

        def body(carry, _):
            st, k, alive = carry
            k, sub = jax.random.split(k)
            act, _ = policy(st.obs, sub)
            nst = jax.vmap(so_env.step, in_axes=(0, 0, 0))(st, act, batched_model)
            step_reward = nst.reward * alive  # zero after termination
            alive = alive * (1.0 - nst.done)
            return (nst, k, alive), step_reward

        init = (state, key, jnp.ones(num_envs))
        _, rewards = jax.lax.scan(body, init, (), length=n_steps)
        return rewards  # (n_steps, num_envs)

    rng_reset = jax.random.split(jax.random.PRNGKey(seed + 1), num_envs)
    rewards = np.asarray(rollout(rng_reset, jax.random.PRNGKey(seed + 2)))
    return designs, rewards
