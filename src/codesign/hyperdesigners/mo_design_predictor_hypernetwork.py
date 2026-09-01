"""``mo_design_hypernetwork`` training algo using design predictor to guide designs
"""

import functools
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import gradients
from brax.training.acme import running_statistics
from brax.training.types import Params

from codesign.hyperdesigners import networks as net_lib
from codesign.hyperdesigners import shared
from codesign.utils import model as model_lib
from codesign.utils.grid import Grid, sample_tradeoffs_cpu
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    DesignPredictorTransition,
    compute_mo_design_hypernet_loss,
    compute_grpo_loss,
)


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
    value_loss_type: str = "mse",
    huber_delta: float = 1.0,
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
    num_cells = num_tradeoffs * num_designs
    assert num_envs % num_cells == 0, (
        "num_envs must be divisible by num_tradeoffs * num_designs"
    )
    assert num_eval_envs % (num_eval_tradeoffs * num_eval_designs) == 0, (
        "num_eval_envs must be divisible by num_eval_tradeoffs * num_eval_designs"
    )
    envs_per_cell = num_envs // num_cells
    eval_envs_per_cell = num_eval_envs // (num_eval_designs * num_eval_tradeoffs)
    schedule = shared.Schedule.make(
        num_timesteps, num_evals, num_envs, batch_size, num_minibatches,
        unroll_length, resamples_per_epoch,
    )
    assert 0 <= num_warmup_iters <= schedule.num_evals_after_init, (
        "num_warmup_iters must be in [0, num_evals - 1]"
    )
    if design_clipping_epsilon is None:
        design_clipping_epsilon = clipping_epsilon

    key = jax.random.PRNGKey(seed)
    key, key_net, key_design_net = jax.random.split(key, 3)
    tradeoff_rng = np.random.default_rng(seed + 1)
    warmup_rng = np.random.default_rng(seed + 2)

    num_objectives = len(environment.params.reward.optimization.objectives)
    normalize = (
        running_statistics.normalize if normalize_observations else (lambda x, y: x)
    )
    design_networks = network_factory(
        observation_size=environment.observation_size,
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
    make_policy = lambda norm, params, designs, tradeoffs, **kw: inference_fn(
        (norm, params.hypernetwork), designs, tradeoffs, **kw
    )

    optimizer = shared.make_optimizer(learning_rate, max_grad_norm)
    loss_fn = functools.partial(
        compute_mo_design_hypernet_loss,
        design_networks       = design_networks,
        entropy_cost          = entropy_cost,
        discounting           = discounting,
        reward_scaling        = reward_scaling,
        gae_lambda            = gae_lambda,
        clipping_epsilon      = clipping_epsilon,
        normalize_advantage   = normalize_advantage,
        value_loss_type       = value_loss_type,
        huber_delta           = huber_delta,
    )
    chunk = shared.make_training_chunk(
        environment, make_policy,
        shared.make_sgd_step(loss_fn, optimizer, num_minibatches),
        schedule, unroll_length, episode_length, num_updates_per_batch,
    )
    rollout_returns = shared.make_rollout_returns(
        environment, make_policy, episode_length, deterministic_eval
    )
    env_inputs = shared.make_env_inputs(environment)

    # ------------------------------------------------------------ design predictor f
    design_optimizer = shared.make_optimizer(design_learning_rate, max_grad_norm)
    grpo_loss_fn = functools.partial(
        compute_grpo_loss,
        design_networks       = design_networks,
        entropy_cost          = design_entropy_cost,
        clipping_epsilon      = design_clipping_epsilon,
    )
    design_gradient_update_fn = gradients.gradient_update_fn(
        grpo_loss_fn, design_optimizer, pmap_axis_name=None, has_aux=True
    )

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

    discounts = discounting ** jnp.arange(episode_length)

    def discounted_scalarized(rewards, tradeoffs):
        """Per-env discounted, tradeoff-scalarized return: ``sum_t gamma^t (w . r_t)``."""
        scalarized = jnp.sum(rewards * tradeoffs[None], axis=-1)  # [T, num_envs]
        return jnp.sum(scalarized * discounts[:, None], axis=0)   # [num_envs]

    # ------------------------------------------------------------------ algo hooks

    def sample(it, design_state, key):
        """Designs paired to their tradeoff: space-filling while ``f`` is frozen, else
        sampled from ``f(d | w)``."""
        if it < num_warmup_iters:
            tradeoffs = sample_tradeoffs_cpu(
                tradeoff_rng, num_tradeoffs, num_objectives,
                sampling=sampling, alpha=alpha,
            )
            low, high = np.asarray(environment.design_limits)
            designs = model_lib.sample_designs(
                warmup_rng, num_designs * num_tradeoffs, low, high, design_dim
            ).reshape(num_designs, num_tradeoffs, design_dim)
            return Grid.paired(
                designs, tradeoffs, envs_per_cell, objectives=environment.objectives
            ), None
        return Grid.from_predictor(
            environment,
            design_predictor_inference_fn,
            design_state.params,
            seed          = tradeoff_rng,
            n_tradeoffs   = num_tradeoffs,
            n_designs     = num_designs,
            per_cell      = envs_per_cell,
            sampling      = sampling,
            alpha         = alpha,
            key           = key,
            deterministic = False,
        )

    def post_chunk(training_state, design_state, sampled, key):
        """One GRPO update of ``f(d | w)`` on the designs this chunk just trained."""
        key_value, key_grad = jax.random.split(key)
        # GRPO value: one deterministic episode per env under the just-trained policy, so
        # every design in a group is judged against the same H(d, w).
        values = discounted_scalarized(
            rollout_returns(
                training_state.normalizer_params,
                training_state.params,
                sampled.designs, sampled.tradeoffs, sampled.model,
                jax.random.split(key_value, num_envs), key_value,
            ),
            sampled.tradeoffs,
        )
        # [num_envs] -> (G, n_tradeoffs, per_cell) -> average out the repetitions,
        # then to the tradeoff-major group layout GRPO reads.
        values = jnp.mean(sampled.grid.unflatten(values), axis=2).swapaxes(0, 1)

        # Warmup designs did not come from f, so there is no ratio to update it on.
        design_metrics = {}
        if sampled.aux is not None:
            design_state, design_metrics = design_sgd(
                design_state,
                DesignPredictorTransition(
                    tradeoff        = jnp.asarray(sampled.grid.unique_tradeoffs),
                    raw_design      = sampled.aux["raw_action"],
                    value           = values,
                    design_log_prob = sampled.aux["log_prob"],
                ),
                key_grad,
            )
        return design_state, {
            **{f"design_{k}": v for k, v in design_metrics.items()},
            "design_value": jnp.mean(values),
        }

    # Tradeoffs are pinned across epochs so hypervolume stays comparable; only the
    # designs move, tracking the predictor.
    eval_tradeoffs = sample_tradeoffs_cpu(
        np.random.default_rng(seed + 1001), num_eval_tradeoffs, num_objectives,
        sampling=sampling, alpha=alpha,
    )
    corner_tradeoffs = jnp.eye(num_objectives, dtype=jnp.float32)

    def evaluate(training_state, design_state, key):
        key_grid, key_rollout = jax.random.split(key)
        grid, _ = Grid.from_predictor(
            environment,
            design_predictor_inference_fn,
            design_state.params,
            seed          = None,
            n_tradeoffs   = num_eval_tradeoffs,
            n_designs     = num_eval_designs,
            per_cell      = eval_envs_per_cell,
            tradeoffs     = eval_tradeoffs,
            key           = key_grid,
            deterministic = False,
        )
        model, designs, tradeoffs = env_inputs(grid)
        rewards = rollout_returns(
            training_state.normalizer_params,
            training_state.params,
            designs, tradeoffs, model,
            jax.random.split(key_rollout, num_eval_envs), key_rollout,
        )
        metrics = shared.eval_metrics(jnp.sum(rewards, axis=0), grid)

        # The predictor's mode design at each simplex corner: does f separate the extremes?
        corner_designs = model_lib.unnormalize_design(
            design_predictor_inference_fn(
                design_state.params, corner_tradeoffs, deterministic=True
            )[0],
            *np.asarray(environment.design_limits),
        )
        for i in range(num_objectives):
            for j in range(design_dim):
                metrics[f"eval/design_mode_obj{i}_dim{j}"] = float(corner_designs[i, j])
        return metrics

    params_of = lambda ts, design_state: (
        ts.normalizer_params, ts.params.hypernetwork, design_state.params
    )

    # Initialize training state.
    training_state = shared.init_training_state(
        DesignHypernetParams(hypernetwork=design_networks.hypernetwork.init(key_net)),
        optimizer, environment.observation_size,
    )
    design_init_params = design_networks.design_predictor_network.init(key_design_net)
    design_predictor_state = DesignPredictorTrainingState(
        optimizer_state=design_optimizer.init(design_init_params),
        params=design_init_params,
    )

    if num_timesteps == 0:
        return inference_fn, params_of(training_state, design_predictor_state), {}

    params, metrics = shared.run_training(
        shared.Algorithm(
            sample        = sample,
            chunk         = chunk,
            evaluate      = evaluate,
            params_of     = params_of,
            post_chunk    = post_chunk,
            epoch_metrics = lambda it: {"training/warmup": float(it < num_warmup_iters)},
        ),
        schedule,
        training_state,
        environment,
        num_envs,
        key,
        inference_fn,
        env_inputs,
        extra_state=design_predictor_state,
        num_evals=num_evals,
        run_evals=run_evals,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    return inference_fn, params, metrics
