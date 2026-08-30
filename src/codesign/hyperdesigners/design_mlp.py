"""``design_mlp`` training algo.

A plain policy/value MLP pair conditioned on the design by appending it to the
observation, rather than by a hypernetwork generating the weights from it.
"""

import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics

from codesign.hyperdesigners import networks as net_lib
from codesign.hyperdesigners import shared
from codesign.utils.grid import Grid
from codesign.hyperdesigners.losses import (
    DesignMLPParams,
    compute_design_mlp_loss,
)


def train_design_mlp(
    environment,
    num_timesteps: int,
    episode_length: int,
    num_envs: int = 128,
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
    num_designs: int = 8,
    resamples_per_epoch: int = 1,
    network_factory: Callable = net_lib.make_design_mlp_networks,
    num_evals: int = 10,
    num_eval_envs: int = 64,
    deterministic_eval: bool = True,
    seed: int = 0,
    progress_fn: Callable = lambda *a: None,
    policy_params_fn: Callable = lambda *a: None,
    run_evals: bool = True,
    # Accepted for compatibility with minimal-mjx's train (which calls train_fn with
    # these); unused here because this env is model-as-input with its own acting/eval.
    wrap_env_fn: Callable | None = None,
    eval_env=None,
):
    assert num_envs % num_designs == 0, (
        "num_envs must be divisible by num_designs"
    )
    assert num_eval_envs % num_designs == 0, (
        "num_eval_envs must be divisible by num_designs"
    )
    schedule = shared.Schedule.make(
        num_timesteps, num_evals, num_envs, batch_size, num_minibatches,
        unroll_length, resamples_per_epoch,
    )

    key = jax.random.PRNGKey(seed)
    key, key_net = jax.random.split(key)
    design_rng = np.random.default_rng(seed)

    # The design rides along in the observation, so every observation the networks and the
    # normalizer see is design_dim wider than the env's own.
    obs_size = jax.tree_util.tree_map(
        lambda size: (
            size[:-1] + (size[-1] + design_dim,)
            if isinstance(size, tuple)
            else size + design_dim
        ),
        environment.observation_size,
        is_leaf=lambda size: isinstance(size, tuple),
    )

    def append_design(data):
        return jax.tree_util.tree_map(
            lambda obs: jnp.concatenate((obs, data.design), axis=-1), data.observation
        )

    normalize = (
        running_statistics.normalize if normalize_observations else (lambda x, y: x)
    )
    design_networks = network_factory(
        observation_size=obs_size,
        action_size=environment.action_size,
        design_dim=design_dim,
        key=key_net,
        preprocess_observations_fn=normalize,
    )
    inference_fn = net_lib.make_design_mlp_inference_fn(design_networks)
    # The policy appends the design itself, and the trivial tradeoff the grid carries goes
    # unused.
    make_policy = lambda norm, params, designs, tradeoffs, **kw: inference_fn(
        (norm, params.policy_params), designs, **kw
    )

    optimizer = shared.make_optimizer(learning_rate, max_grad_norm)
    loss_fn = functools.partial(
        compute_design_mlp_loss,
        design_networks       = design_networks,
        entropy_cost          = entropy_cost,
        discounting           = discounting,
        reward_scaling        = reward_scaling,
        gae_lambda            = gae_lambda,
        clipping_epsilon      = clipping_epsilon,
        normalize_advantage   = normalize_advantage,
    )
    chunk = shared.make_training_chunk(
        environment, make_policy,
        shared.make_sgd_step(loss_fn, optimizer, num_minibatches),
        schedule, unroll_length, episode_length, num_updates_per_batch,
        observation_fn=append_design,
    )
    rollout_returns = shared.make_rollout_returns(
        environment, make_policy, episode_length, deterministic_eval
    )
    env_inputs = shared.make_env_inputs(environment)

    def sample(it, extra_state, key):
        """``num_designs`` designs, tiled across the envs, against the trivial tradeoff."""
        return Grid.from_design_sample(
            environment, design_rng, num_designs, per_cell=num_envs // num_designs
        ), None

    # Held fixed across evals, so returns are comparable epoch to epoch.
    eval_grid = Grid.from_design_sample(
        environment, seed + 1000, num_designs, per_cell=num_eval_envs // num_designs
    )
    eval_model, eval_designs, eval_tradeoffs = env_inputs(eval_grid)

    def evaluate(training_state, extra_state, key):
        rewards = rollout_returns(
            training_state.normalizer_params,
            training_state.params,
            eval_designs,
            eval_tradeoffs,
            eval_model,
            jax.random.split(key, num_eval_envs),
            key,
        )
        return shared.eval_metrics(jnp.sum(rewards, axis=0), eval_grid)

    params_of = lambda ts, extra: (ts.normalizer_params, ts.params.policy_params)

    key_policy, key_value = jax.random.split(key_net)
    training_state = shared.init_training_state(
        DesignMLPParams(
            policy_params=design_networks.policy_network.init(key_policy),
            value_params=design_networks.value_network.init(key_value),
        ),
        optimizer, obs_size,
    )
    if num_timesteps == 0:
        return inference_fn, params_of(training_state, None), {}

    params, metrics = shared.run_training(
        shared.Algorithm(
            sample=sample, chunk=chunk, evaluate=evaluate, params_of=params_of
        ),
        schedule,
        training_state,
        environment,
        num_envs,
        key,
        inference_fn,
        env_inputs,
        num_evals=num_evals,
        run_evals=run_evals,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    return inference_fn, params, metrics
