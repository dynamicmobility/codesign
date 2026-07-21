"""Parallelized rollouts of a model-specific policy across a sweep of Codesign Envs.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from codesign.envs.CodesignBase import CodesignMO2SO, CodesignBase, MOCodesignBase
from minimal_mjx.eval import policy as policy_lib
from codesign.hyperdesigners import acting
from codesign.learning.inference import load_design_hypernetwork, load_mo_design_hypernetwork
from codesign.utils import model as model_lib
from codesign.utils.grid import DesignTradeoffSampleGrid, DesignTradeoffRolloutGrid


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

def build_grid_rollout_fn(env, n_steps, make_policy, deterministic=True):
    """Build a jitted rollout of a design/tradeoff-conditioned hypernetwork policy,
    vmapped over grid axes (design, tradeoff, rep).

    The returned function has signature ``rollout(keys, designs, tradeoffs, models,
    params)`` with ``keys`` of shape ``(n_designs, n_tradeoffs, per_cell, 2)``,
    ``designs`` ``(n_designs, design_dim)``, ``tradeoffs`` ``(n_tradeoffs, num_objectives)``
    and ``models`` batched per design. Outputs keep the leading
    ``(n_designs, n_tradeoffs, per_cell)`` grid axes.
    """
    policy_rng = jax.random.PRNGKey(0)

    def step_fn(carry, _, model, policy):
        state, old_reward = carry
        action, _ = policy(state.obs, policy_rng)
        new_state = env.step(state, action, model)
        # accumulate per-objective reward, masking out steps after termination
        return (new_state, old_reward + new_state.reward * (1 - new_state.done)), new_state

    def rollout(key, design, tradeoff, model, params):
        policy = make_policy(
            params         = params,
            designs        = design,
            tradeoffs      = tradeoff,
            deterministic  = deterministic,
        )
        scan_step_fn = functools.partial(step_fn, model=model, policy=policy)
        state = env.reset(key, model)
        carry = (state, state.reward)
        return jax.lax.scan(scan_step_fn, carry, (), n_steps)

    over_reps      = jax.vmap(rollout,    in_axes=(0, None, None, None, None))
    over_tradeoffs = jax.vmap(over_reps,  in_axes=(0, None, 0, None, None))
    over_designs   = jax.vmap(over_tradeoffs, in_axes=(0, 0, None, 0, None))
    return jax.jit(over_designs)

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
) -> DesignTradeoffRolloutGrid:
    """Roll out a trained MO design hypernetwork over a design x tradeoff grid.

    Designs are a uniform sweep of the configured design range; tradeoffs are Dirichlet
    samples on the objective simplex. Each grid cell is rolled out ``per_cell`` times
    for ``n_steps`` env steps.

    Args:
        env: a model-as-input env (e.g. ``CodesignCheetah``).
        config: the run config dict (as written to ``config.yaml`` at train time).
        n_designs: number of designs in the uniform sweep.
        n_tradeoffs: number of sampled tradeoffs.
        per_cell: rollout repetitions per (design, tradeoff) cell.
        n_steps: rollout length in env steps.
        checkpoint_path: explicit checkpoint dir; defaults to latest under ``save_dir/name``.
        seed: base PRNG seed for resets and tradeoff sampling.
        deterministic: take the policy mode (vs. sampling) at each step.

    Returns:
        A :class:`DesignTradeoffRolloutGrid` with ``rewards`` of shape
        ``(n_designs, n_tradeoffs, per_cell, num_objectives)``.
    """
    # Create the sample grid
    grid = DesignTradeoffSampleGrid.from_uniform_sample(
        env, seed=seed, n_tradeoffs=n_tradeoffs, n_designs=n_designs, per_cell=per_cell
    )
    designs_input = model_lib.normalize_design(jnp.asarray(grid.designs), config=config)
    batched_model = grid.build_models(env, tiled=False)

    # Create the rollout function
    make_policy_fn, params = load_mo_design_hypernetwork(config, path=checkpoint_path)
    keys = jax.random.split(
        jax.random.PRNGKey(seed), grid.num_envs
    ).reshape(n_designs, n_tradeoffs, per_cell, -1)
    rollout_fn = build_grid_rollout_fn(
        env           = env,
        n_steps       = n_steps,
        make_policy   = make_policy_fn,
        deterministic = deterministic
    )
    
    # Run the rollouts
    (_, final_rewards), _ = rollout_fn(keys, designs_input, jnp.asarray(grid.tradeoffs), batched_model, params)

    return DesignTradeoffRolloutGrid(
        designs       = grid.designs,
        tradeoffs     = grid.tradeoffs,
        rewards       = np.asarray(final_rewards),
        objectives    = env.objectives,
    )



def rollout_design_hypernetwork(
    env: CodesignBase | MOCodesignBase,
    config,
    num_envs: int,
    n_steps: int,
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

    design = config["learning_params"]["design_params"]
    design_low = float(design["design_low"])
    design_high = float(design["design_high"])
    design_dim = int(design["design_dim"])

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
        n_steps,
        mask_after_done=True,
        seed=seed,
    )
    rewards = rewards.reshape(n_steps, num_envs, trials_per_env).transpose(2, 0, 1)
    return designs, rewards
