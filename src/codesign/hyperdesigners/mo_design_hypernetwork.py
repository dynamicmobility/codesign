"""``mo_design_hypernetwork`` training algo.
"""

import functools
import time
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import gradients
from brax.training.acme import running_statistics
from mujoco import mjx

from codesign.hyperdesigners import acting
from codesign.hyperdesigners import networks as net_lib
from codesign.utils import model as model_lib
from codesign.utils.grid import DesignTradeoffSampleGrid, DesignTradeoffRolloutGrid
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    compute_mo_design_hypernet_loss,
)




@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: DesignHypernetParams
    normalizer_params: running_statistics.RunningStatisticsState


def sample_tradeoffs(
    rng: np.random.Generator,
    it: int,
    num_tradeoffs: int,
    num_objectives: int,
    sampling: str = "dense",
    alpha: float = 1.0,
) -> np.ndarray:
    """Sample ``num_tradeoffs`` simplex tradeoffs (host-side numpy), MORLAX-style.

    ``num_tradeoffs`` is the sole driver of how many distinct tradeoffs are produced.
    Ports the sampling styles of ``morlax.sample_preferences``:
      * ``dense`` — ``num_tradeoffs`` Dirichlet(alpha) draws;
      * ``sparse-heavytail`` — ``num_tradeoffs - num_objectives`` Dirichlet draws plus the
        ``num_objectives`` axis-aligned (one-hot) extreme tradeoffs (e.g. ``num_tradeoffs=8``,
        ``num_objectives=3`` -> 5 simplex draws + 3 one-hot corners);
      * ``single-avg`` — every tradeoff is the uniform ``1/M``.
    During warmup (``it < round(warmup_frac * num_warmup_ref)``) all tradeoffs are uniform.
    Returns an array of shape ``(num_tradeoffs, num_objectives)``.
    """

    if sampling == "dense":
        w = rng.dirichlet(np.ones(num_objectives) * alpha, size=num_tradeoffs)
    elif sampling == "sparse-heavytail":
        n_dir = max(num_tradeoffs - num_objectives, 0)
        dir_w = rng.dirichlet(np.ones(num_objectives) * alpha, size=n_dir)
        w = np.concatenate([dir_w, np.eye(num_objectives)], axis=0)
        w = w[:num_tradeoffs]
    elif sampling == "single-avg":
        w = np.full((num_tradeoffs, num_objectives), 1.0 / num_objectives)
    else:
        raise ValueError(f"Sampling type {sampling} not implemented")
    return w.astype(np.float32)


def train_mo_design_hypernetwork(
    environment,
    num_timesteps: int,
    episode_length: int,
    num_envs: int = 1024,
    num_designs: int = 8,
    num_tradeoffs: int = 8,
    unroll_length: int = 20,
    batch_size: int = 64,
    num_minibatches: int = 2,
    num_updates_per_batch: int = 4,
    learning_rate: float = 5e-4,
    entropy_cost: float = 1e-2,
    discounting: float = 0.98,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.1,
    gae_lambda: float = 0.95,
    max_grad_norm: float | None = 1.0,
    normalize_advantage: bool = True,
    normalize_observations: bool = True,
    design_low: float = 0.5,
    design_high: float = 2.0,
    design_dim: int = 1,
    resamples_per_epoch: int = 1,
    # tradeoff sampling
    alpha: float = 1.0,
    sampling: str = "dense",
    warmup_frac: float = 0.0,
    network_factory: Callable = net_lib.make_mo_design_hypernet_networks,
    num_evals: int = 10,
    num_eval_envs: int = 64,
    deterministic_eval: bool = True,
    seed: int = 0,
    progress_fn: Callable = lambda *a: None,
    policy_params_fn: Callable = lambda *a: None,
    run_evals: bool = True,
    # Accepted for compatibility with minimal-mjx's train (unused here).
    wrap_env_fn: Callable | None = None,
    eval_env=None,
):
    num_cells = num_designs * num_tradeoffs
    assert num_envs % num_cells == 0, (
        "num_envs must be divisible by num_designs * num_tradeoffs"
    )
    assert num_eval_envs % num_cells == 0, (
        "num_eval_envs must be divisible by num_designs * num_tradeoffs"
    )
    assert (batch_size * num_minibatches) % num_envs == 0, (
        "batch_size * num_minibatches must be divisible by num_envs"
    )
    assert resamples_per_epoch >= 1, "resamples_per_epoch must be >= 1"
    envs_per_cell = num_envs // num_cells
    eval_envs_per_cell = num_eval_envs // num_cells
    num_scans = batch_size * num_minibatches // num_envs
    env_step_per_training_step = batch_size * unroll_length * num_minibatches
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_chunk = int(
        np.ceil(
            num_timesteps
            / (num_evals_after_init * resamples_per_epoch * env_step_per_training_step)
        )
    )
    num_training_steps_per_epoch = num_training_steps_per_chunk * resamples_per_epoch

    key = jax.random.PRNGKey(seed)
    key, key_net = jax.random.split(key)
    key_env = jax.random.fold_in(key, 1)
    key_eval = jax.random.fold_in(key, 2)
    design_rng = np.random.default_rng(seed)
    tradeoff_rng = np.random.default_rng(seed + 1)

    jit_reset = jax.jit(
        lambda rngs, model: acting.reset(environment, rngs, model)
    )

    model_treedef = None
    def build_grid(
        n_designs, n_tradeoffs, per_cell, d_rng, w_rng, it, num_objectives
    ):
        """Sample a design x tradeoff grid and build the tiled per-env model.

        Returns ``(grid, batched_model, designs_input, tradeoffs_full)`` where the last
        three have a leading env axis of ``grid.num_envs`` in the grid's flat env ordering.
        """
        nonlocal model_treedef
        designs_unique = model_lib.sample_designs(
            d_rng, n_designs, design_low, design_high, design_dim
        )
        tradeoffs_unique = sample_tradeoffs(
            w_rng, it, n_tradeoffs, num_objectives,
            sampling=sampling, alpha=alpha,
            num_warmup_ref=num_evals_after_init,
        )
        grid = DesignTradeoffSampleGrid(
            designs=designs_unique, tradeoffs=tradeoffs_unique, per_cell=per_cell
        )

        batched_model = grid.build_models(environment, tiled=True)
        leaves, treedef = jax.tree_util.tree_flatten(batched_model)
        if model_treedef is None:
            model_treedef = treedef
        batched_model = jax.tree_util.tree_unflatten(model_treedef, leaves)

        designs_full, tradeoffs_full = grid.flatten()
        designs_input = model_lib.normalize_design(
            jnp.asarray(designs_full), design_low, design_high
        )
        return grid, batched_model, designs_input, jnp.asarray(tradeoffs_full)
    
    obs_size = environment.observation_size
    num_objectives = len(environment.params.reward.optimization.objectives)

    normalize = (
        running_statistics.normalize if normalize_observations else (lambda x, y: x)
    )
    design_networks = network_factory(
        observation_size=obs_size,
        action_size=environment.action_size,
        design_dim=design_dim,
        num_objectives=num_objectives,
        key=key_net,
        preprocess_observations_fn=normalize,
    )
    inference_fn = net_lib.make_mo_design_inference_fn(design_networks)

    optimizer = optax.adam(learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
        )

    loss_fn = functools.partial(
        compute_mo_design_hypernet_loss,
        design_networks       = design_networks,
        entropy_cost          = entropy_cost,
        discounting           = discounting,
        reward_scaling        = reward_scaling,
        gae_lambda            = gae_lambda,
        clipping_epsilon      = clipping_epsilon,
        normalize_advantage   = normalize_advantage,
    )
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=None, has_aux=True
    )

    def minibatch_step(carry, data, normalizer_params):
        opt_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, opt_state = gradient_update_fn(
            params, normalizer_params, data, key_loss, optimizer_state=opt_state
        )
        return (opt_state, params, key), metrics

    def sgd_step(carry, unused_t, data, normalizer_params):
        opt_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert(x):
            x = jax.random.permutation(key_perm, x)
            return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

        shuffled = jax.tree_util.tree_map(convert, data)
        (opt_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (opt_state, params, key_grad),
            shuffled,
            length=num_minibatches,
        )
        return (opt_state, params, key), metrics

    def training_step(
        carry, unused_t, batched_model, designs, tradeoffs, first_state
    ):
        training_state, state, key = carry
        key_sgd, key_unroll, new_key = jax.random.split(key, 3)
        policy = inference_fn(
            (training_state.normalizer_params, training_state.params.hypernetwork),
            designs,
            tradeoffs,
        )

        def scan_unroll(c, _):
            cur_state, cur_key = c
            cur_key, nk = jax.random.split(cur_key)
            nstate, data = acting.mo_generate_unroll(
                environment,
                cur_state,
                batched_model,
                policy,
                designs,
                tradeoffs,
                cur_key,
                unroll_length,
                first_state,
                episode_length,
                extra_fields=(),
            )
            return (nstate, nk), data

        (state, _), data = jax.lax.scan(
            scan_unroll, (state, key_unroll), (), length=num_scans
        )
        # data: [num_scans, unroll_length, num_envs, ...] -> [B, T, ...].
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )

        normalizer_params = running_statistics.update(
            training_state.normalizer_params, data.observation
        )
        (opt_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )
        new_ts = TrainingState(
            optimizer_state=opt_state,
            params=params,
            normalizer_params=normalizer_params,
        )
        return (new_ts, state, new_key), metrics

    @jax.jit
    def training_chunk(
        training_state, state, key, batched_model, designs, tradeoffs, first_state
    ):
        """Train for one resample chunk against a fixed design x tradeoff grid."""
        step = functools.partial(
            training_step,
            batched_model=batched_model,
            designs=designs,
            tradeoffs=tradeoffs,
            first_state=first_state,
        )
        (training_state, state, _), metrics = jax.lax.scan(
            step, (training_state, state, key), (), length=num_training_steps_per_chunk
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, state, metrics

    @jax.jit
    def eval_unroll(
        normalizer_params, hypernet_params, designs, tradeoffs, batched_model, rngs, key
    ):
        state = acting.reset(environment, rngs, batched_model)
        policy = inference_fn(
            (normalizer_params, hypernet_params),
            designs,
            tradeoffs,
            deterministic=deterministic_eval,
        )

        def body(carry, _):
            st, k, alive, ret = carry
            k, sub = jax.random.split(k)
            act, _ = policy(st.obs, sub)
            nst = jax.vmap(environment.step, in_axes=(0, 0, 0))(st, act, batched_model)
            # Accumulate the per-objective reward vector while the episode is alive.
            ret = ret + nst.reward * alive[:, None]
            alive = alive * (1.0 - nst.done)
            return (nst, k, alive, ret), None

        init = (
            state,
            key,
            jnp.ones(num_eval_envs),
            jnp.zeros((num_eval_envs, num_objectives)),
        )
        (_, _, _, ret), _ = jax.lax.scan(body, init, (), length=episode_length)
        return ret

    eval_grid_bundle = build_grid(
        num_designs, num_tradeoffs, eval_envs_per_cell,
        np.random.default_rng(seed + 1000),
        np.random.default_rng(seed + 1001),
        num_evals_after_init,  # past warmup for eval
        num_objectives,
    )

    def evaluate(training_state, key):
        eval_grid, eval_model, eval_designs, eval_tradeoffs = eval_grid_bundle
        eval_rngs = jax.random.split(key, num_eval_envs)
        ret = eval_unroll(
            training_state.normalizer_params,
            training_state.params.hypernetwork,
            eval_designs,
            eval_tradeoffs,
            eval_model,
            eval_rngs,
            key,
        )
        ret = np.asarray(ret)  # [num_eval_envs, num_objectives]
        tradeoffs = np.asarray(eval_tradeoffs)
        scalarized = np.sum(tradeoffs * ret, axis=-1)
        metrics = {
            "eval/episode_reward": float(np.mean(scalarized)),
            "eval/episode_reward_std": float(np.std(scalarized)),
        }
        for i in range(num_objectives):
            metrics[f"eval/episode_reward_obj{i}"] = float(np.mean(ret[:, i]))

        # Design x tradeoff grid of per-objective returns, for Pareto plotting.
        metrics["eval_grid"] = DesignTradeoffRolloutGrid.from_flat(
            eval_grid, ret, objectives=environment.objectives
        )
        return metrics

    # Initialize training state.
    init_params = DesignHypernetParams(
        hypernetwork=design_networks.hypernetwork.init(key_net)
    )
    normalizer_params = running_statistics.init_state(
        model_lib.observation_spec(obs_size)
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=normalizer_params,
    )

    if num_timesteps == 0:
        return (
            inference_fn,
            (training_state.normalizer_params, training_state.params.hypernetwork),
            {},
        )

    # Initial eval + checkpoint.
    metrics = {}
    if run_evals and num_evals > 1:
        metrics = evaluate(training_state, key_eval)
        progress_fn(0, metrics)
    params = (training_state.normalizer_params, training_state.params.hypernetwork)
    policy_params_fn(0, inference_fn, params)

    walltime = 0.0
    for it in range(num_evals_after_init):
        t0 = time.time()
        chunk_metrics = []
        for _ in range(resamples_per_epoch):
            # Redraw the grid, rebuild the per-env models, and restart the envs on them
            # (the robot itself changed, so the carried state is stale).
            _, batched_model, designs_input, tradeoffs = build_grid(
                num_designs, num_tradeoffs, envs_per_cell,
                design_rng, tradeoff_rng, it, num_objectives,
            )
            key_env, sub = jax.random.split(key_env)
            rngs = jax.random.split(sub, num_envs)
            env_state = jit_reset(rngs, batched_model)
            first_state = env_state

            key, chunk_key = jax.random.split(key)
            training_state, env_state, train_metrics = training_chunk(
                training_state, env_state, chunk_key, batched_model,
                designs_input, tradeoffs, first_state,
            )
            chunk_metrics.append(train_metrics)

        train_metrics = jax.tree_util.tree_map(
            lambda *xs: jnp.mean(jnp.stack(xs)), *chunk_metrics
        )
        train_metrics = jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), train_metrics
        )
        epoch_time = time.time() - t0
        walltime += epoch_time
        current_step = (
            (it + 1) * num_training_steps_per_epoch * env_step_per_training_step
        )

        metrics = {
            "training/sps": (num_training_steps_per_epoch * env_step_per_training_step)
            / epoch_time,
            "training/walltime": walltime,
            **{f"training/{k}": float(v) for k, v in train_metrics.items()},
        }
        if run_evals:
            key_eval, eval_subkey = jax.random.split(key_eval)
            metrics.update(evaluate(training_state, eval_subkey))

        params = (training_state.normalizer_params, training_state.params.hypernetwork)
        policy_params_fn(current_step, inference_fn, params)
        progress_fn(current_step, metrics)

    return inference_fn, params, metrics
