"""``mo_design_hypernetwork`` training algo using design predictor to guide designs
"""

import dataclasses
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
from brax.training.types import Params

from codesign.hyperdesigners import acting
from codesign.hyperdesigners import networks as net_lib
from codesign.utils import model as model_lib
from codesign.utils.grid import Grid, sample_tradeoffs
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    DesignPredictorTransition,
    compute_mo_design_hypernet_loss,
    compute_grpo_loss,
)


@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: DesignHypernetParams
    normalizer_params: running_statistics.RunningStatisticsState


@flax.struct.dataclass
class DesignPredictorTrainingState:
    """The design predictor's input is a simplex tradeoff, so it needs no normalizer."""

    optimizer_state: optax.OptState
    params: Params


def train_mo_design_predictor(
    environment,
    num_timesteps: int,
    episode_length: int,
    num_envs: int = 1024,
    num_designs: int = 8,  # GRPO group size: designs drawn per tradeoff
    num_eval_designs: int = 8,
    num_tradeoffs: int = 8,
    num_eval_tradeoffs: int = 8,
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
    design_dim: int = 1,
    resamples_per_epoch: int = 1,
    # design predictor
    design_learning_rate: float = 1e-3,
    design_entropy_cost: float = 1e-3,
    design_clipping_epsilon: float | None = None,  # None follows the policy's
    num_design_updates_per_batch: int = 4,
    num_warmup_iters: int = 0,  # leading epochs with space-filling designs and f frozen
    # tradeoff sampling
    alpha: float = 1.0,
    sampling: str = "dense",
    network_factory: Callable = net_lib.make_mo_design_predictor_hypernet_networks,
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
    group_size = num_designs  # GRPO group size G
    num_cells = num_tradeoffs * group_size
    assert num_envs % num_cells == 0, (
        "num_envs must be divisible by num_tradeoffs * num_designs"
    )
    assert num_eval_envs % (num_eval_tradeoffs * num_eval_designs) == 0, (
        "num_eval_envs must be divisible by num_eval_tradeoffs * num_eval_designs"
    )
    assert (batch_size * num_minibatches) % num_envs == 0, (
        "batch_size * num_minibatches must be divisible by num_envs"
    )
    assert resamples_per_epoch >= 1, "resamples_per_epoch must be >= 1"
    envs_per_cell = num_envs // num_cells
    eval_envs_per_cell = num_eval_envs // (num_eval_designs * num_eval_tradeoffs)
    num_scans = batch_size * num_minibatches // num_envs
    env_step_per_training_step = batch_size * unroll_length * num_minibatches
    num_evals_after_init = max(num_evals - 1, 1)
    assert 0 <= num_warmup_iters <= num_evals_after_init, (
        "num_warmup_iters must be in [0, num_evals - 1]"
    )
    if design_clipping_epsilon is None:
        design_clipping_epsilon = clipping_epsilon
    num_training_steps_per_chunk = int(
        np.ceil(
            num_timesteps
            / (num_evals_after_init * resamples_per_epoch * env_step_per_training_step)
        )
    )
    num_training_steps_per_epoch = num_training_steps_per_chunk * resamples_per_epoch

    key = jax.random.PRNGKey(seed)
    key, key_net, key_design_net = jax.random.split(key, 3)
    key_env = jax.random.fold_in(key, 1)
    key_eval = jax.random.fold_in(key, 2)
    key_design = jax.random.fold_in(key, 3)
    tradeoff_rng = np.random.default_rng(seed + 1)
    warmup_rng = np.random.default_rng(seed + 2)

    jit_reset = jax.jit(
        lambda rngs, model: acting.reset(environment, rngs, model)
    )

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
    design_predictor_inference_fn = net_lib.make_design_predictor_inference_fn(
        design_networks
    )

    optimizer = optax.adam(learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
        )

    design_optimizer = optax.adam(design_learning_rate)
    if max_grad_norm is not None:
        design_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(design_learning_rate)
        )

    reference_model = None  # treedef of the first stacked model; see build_grid
    def build_grid(
        n_designs, n_tradeoffs, per_cell, rng, key_sample, it, design_params,
        tradeoffs=None,
    ):
        """Sample a tradeoff-paired design grid from ``f(d | w)`` and build its models."""
        nonlocal reference_model
        if it < num_warmup_iters:
            # Warmup: cover the box with a Sobol sample instead of drawing from f.
            if tradeoffs is None:
                tradeoffs = sample_tradeoffs(
                    rng, n_tradeoffs, num_objectives, sampling=sampling, alpha=alpha
                )
            low, high = np.asarray(environment.design_limits)
            designs = model_lib.sample_designs(
                warmup_rng, n_designs * n_tradeoffs, low, high, design_dim
            ).reshape(n_designs, n_tradeoffs, design_dim)
            grid, extras = Grid.paired(
                designs, tradeoffs, per_cell, objectives=environment.objectives
            ), None
        else:
            grid, extras = Grid.from_predictor(
                environment,
                design_predictor_inference_fn,
                design_params,
                seed          = rng,
                n_tradeoffs   = n_tradeoffs,
                n_designs     = n_designs,
                per_cell      = per_cell,
                sampling      = sampling,
                alpha         = alpha,
                tradeoffs     = tradeoffs,
                key           = key_sample,
                deterministic = False,
            )

        batched_model, designs_input, tradeoffs_full = grid.env_inputs(
            environment, like=reference_model
        )
        reference_model = batched_model
        return grid, batched_model, designs_input, tradeoffs_full, extras

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

    grpo_loss_fn = functools.partial(
        compute_grpo_loss,
        design_networks       = design_networks,
        entropy_cost          = design_entropy_cost,
        clipping_epsilon      = design_clipping_epsilon,
    )
    design_gradient_update_fn = gradients.gradient_update_fn(
        grpo_loss_fn, design_optimizer, pmap_axis_name=None, has_aux=True
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
        # Compute dataset of rollouts
        (state, _), data = jax.lax.scan(
            scan_unroll, (state, key_unroll), (), length=num_scans
        )
        # data: [num_scans, unroll_length, num_envs, ...] -> [B, T, ...].
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        # normalize observations based on current state distribution
        normalizer_params = running_statistics.update(
            training_state.normalizer_params, data.observation
        )
        # Take SGD steps to update policy params
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
    def rollout_returns(
        normalizer_params, hypernet_params, designs, tradeoffs, batched_model, rngs, key
    ):
        """Per-objective reward per env over one episode, zeroed after termination.

        Returns ``[episode_length, num_envs, num_objectives]``. An env that terminates
        early contributes zero from its termination step onward.
        """
        state = acting.reset(environment, rngs, batched_model)
        policy = inference_fn(
            (normalizer_params, hypernet_params),
            designs,
            tradeoffs,
            deterministic=deterministic_eval,
        )

        def body(carry, _):
            st, k, alive = carry
            k, sub = jax.random.split(k)
            act, _ = policy(st.obs, sub)
            nst = jax.vmap(environment.step, in_axes=(0, 0, 0))(st, act, batched_model)
            reward = nst.reward * alive[:, None]
            alive = alive * (1.0 - nst.done)
            return (nst, k, alive), reward

        init = (state, key, jnp.ones(rngs.shape[0]))
        _, rewards = jax.lax.scan(body, init, (), length=episode_length)
        return rewards

    discounts = discounting ** jnp.arange(episode_length)

    def discounted_scalarized(rewards, tradeoffs):
        """Per-env discounted, tradeoff-scalarized return: ``sum_t gamma^t (w . r_t)``."""
        scalarized = jnp.sum(rewards * tradeoffs[None], axis=-1)  # [T, num_envs]
        return jnp.sum(scalarized * discounts[:, None], axis=0)   # [num_envs]

    def design_minibatch_step(carry, unused_t, data):
        opt_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, opt_state = design_gradient_update_fn(
            params, data, key_loss, optimizer_state=opt_state
        )
        return (opt_state, params, key), metrics

    @jax.jit
    def design_sgd(design_state, data, key):
        """``num_design_updates_per_batch`` GRPO epochs over one batch of groups.

        On the first epoch ``rho == 1`` and the clip is inert; from the second on it
        constrains the step, which is why the group data is reused rather than taking a
        single gradient step per resample.
        """
        (opt_state, params, _), metrics = jax.lax.scan(
            functools.partial(design_minibatch_step, data=data),
            (design_state.optimizer_state, design_state.params, key),
            (),
            length=num_design_updates_per_batch,
        )
        new_state = DesignPredictorTrainingState(
            optimizer_state=opt_state, params=params
        )
        return new_state, jax.tree_util.tree_map(jnp.mean, metrics)

    # Tradeoffs are pinned across epochs so hypervolume stays comparable; only the
    # designs move, tracking the predictor.
    eval_tradeoffs = sample_tradeoffs(
        np.random.default_rng(seed + 1001), num_eval_tradeoffs, num_objectives,
        sampling=sampling, alpha=alpha,
    )
    corner_tradeoffs = jnp.eye(num_objectives, dtype=jnp.float32)

    def evaluate(training_state, design_params, key):
        key, key_grid, key_rollout = jax.random.split(key, 3)
        grid, model, designs, tradeoffs, _ = build_grid(
            num_eval_designs, num_eval_tradeoffs, eval_envs_per_cell,
            None, key_grid, num_evals_after_init, design_params,
            tradeoffs=eval_tradeoffs,
        )
        eval_rngs = jax.random.split(key_rollout, num_eval_envs)
        rewards = rollout_returns(
            training_state.normalizer_params,
            training_state.params.hypernetwork,
            designs,
            tradeoffs,
            model,
            eval_rngs,
            key_rollout,
        )
        ret = np.asarray(jnp.sum(rewards, axis=0))  # [num_eval_envs, num_objectives]
        scalarized = np.sum(np.asarray(tradeoffs) * ret, axis=-1)
        metrics = {
            "eval/episode_reward": float(np.mean(scalarized)),
            "eval/episode_reward_std": float(np.std(scalarized)),
        }
        for i in range(num_objectives):
            metrics[f"eval/episode_reward_obj{i}"] = float(np.mean(ret[:, i]))

        # The predictor's mode design at each simplex corner: does f separate the extremes?
        corner_designs = model_lib.unnormalize_design(
            design_predictor_inference_fn(
                design_params, corner_tradeoffs, deterministic=True
            )[0],
            *np.asarray(environment.design_limits),
        )
        for i in range(num_objectives):
            for j in range(design_dim):
                metrics[f"eval/design_mode_obj{i}_dim{j}"] = float(corner_designs[i, j])

        # Predictor sample x tradeoff grid of per-objective returns, for Pareto plotting.
        metrics["eval_grid"] = dataclasses.replace(grid, rewards=grid.unflatten(ret))
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
    design_init_params = design_networks.design_predictor_network.init(key_design_net)
    design_predictor_state = DesignPredictorTrainingState(
        optimizer_state=design_optimizer.init(design_init_params),
        params=design_init_params,
    )

    def checkpoint_params(training_state, design_predictor_state):
        return (
            training_state.normalizer_params,
            training_state.params.hypernetwork,
            design_predictor_state.params,
        )

    if num_timesteps == 0:
        return (
            inference_fn,
            checkpoint_params(training_state, design_predictor_state),
            {},
        )

    # Initial eval + checkpoint.
    metrics = {}
    if run_evals and num_evals > 1:
        metrics = evaluate(training_state, design_predictor_state.params, key_eval)
        progress_fn(0, metrics)
    params = checkpoint_params(training_state, design_predictor_state)
    policy_params_fn(0, inference_fn, params)

    walltime = 0.0
    for it in range(num_evals_after_init):
        t0 = time.time()
        chunk_metrics = []
        for _ in range(resamples_per_epoch):
            # Redraw the designs and restart the envs on them
            key_design, key_sample = jax.random.split(key_design)
            grid, batched_model, designs_input, tradeoffs, extras = build_grid(
                group_size, num_tradeoffs, envs_per_cell,
                tradeoff_rng, key_sample, it, design_predictor_state.params,
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

            # GRPO value: one deterministic episode per env under the just-trained
            # policy, so every design in a group is judged against the same H(d, w).
            key_env, sub = jax.random.split(key_env)
            values = discounted_scalarized(
                rollout_returns(
                    training_state.normalizer_params,
                    training_state.params.hypernetwork,
                    designs_input, tradeoffs, batched_model,
                    jax.random.split(sub, num_envs), sub,
                ),
                tradeoffs,
            )
            # [num_envs] -> (G, n_tradeoffs, per_cell) -> average out the repetitions,
            # then to the tradeoff-major group layout GRPO reads.
            values = jnp.mean(grid.unflatten(values), axis=2).swapaxes(0, 1)

            # Warmup designs did not come from f, so there is no ratio to update it on.
            design_metrics = {}
            if extras is not None:
                key_design, key_grad = jax.random.split(key_design)
                design_predictor_state, design_metrics = design_sgd(
                    design_predictor_state,
                    DesignPredictorTransition(
                        tradeoff        = jnp.asarray(grid.unique_tradeoffs),
                        raw_design      = extras["raw_action"],
                        value           = values,
                        design_log_prob = extras["log_prob"],
                    ),
                    key_grad,
                )
            chunk_metrics.append({
                **train_metrics,
                **{f"design_{k}": v for k, v in design_metrics.items()},
                "design_value": jnp.mean(values),
            })

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
            "training/warmup": float(it < num_warmup_iters),
            **{f"training/{k}": float(v) for k, v in train_metrics.items()},
        }
        if run_evals:
            key_eval, eval_subkey = jax.random.split(key_eval)
            metrics.update(
                evaluate(training_state, design_predictor_state.params, eval_subkey)
            )

        params = checkpoint_params(training_state, design_predictor_state)
        policy_params_fn(current_step, inference_fn, params)
        progress_fn(current_step, metrics)

    return inference_fn, params, metrics
