"""``design_hypernetwork`` training algo.

A single-objective PPO algo that trains a *design-conditioned* hypernetwork 
(policy + separate value hypernetwork) on a model-as-input (MAI) environment. 
Designs are sampled and stacked every training epoch.

v1 simplifications (documented intentionally):
  * single device (``jax.jit``, no ``pmap``);
  * env state is re-sampled each epoch (new designs), so episodes don't span epochs;
  * ``MAICheetah`` resets are deterministic per design (no obs/init randomization).
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
from brax.training.acme import running_statistics, specs
from mujoco import mjx

from codesign.hyperdesigners import acting
from codesign.hyperdesigners import networks as net_lib
from codesign.utils import model as model_lib
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    compute_design_hypernet_loss,
)


@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: DesignHypernetParams
    normalizer_params: running_statistics.RunningStatisticsState


def train_design_hypernetwork(
    environment,
    num_timesteps: int,
    episode_length: int,
    generate_model_fn: Callable[[np.ndarray], mjx.Model] | None = None,
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
    design_low: float = 0.5,
    design_high: float = 2.0,
    design_dim: int = 1,
    reward_objective_weights: tuple | None = None,
    network_factory: Callable = net_lib.make_design_hypernet_networks,
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
    assert (batch_size * num_minibatches) % num_envs == 0, (
        "batch_size * num_minibatches must be divisible by num_envs"
    )
    num_scans = batch_size * num_minibatches // num_envs
    env_step_per_training_step = batch_size * unroll_length * num_minibatches
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = int(
        np.ceil(num_timesteps / (num_evals_after_init * env_step_per_training_step))
    )

    # By default, derive the (host-side, non-jittable) design->model builder from the env.
    if generate_model_fn is None:
        def generate_model_fn(design_row):
            d = float(np.asarray(design_row).reshape(-1)[0])
            return mjx.put_model(environment.generate_model(d))

    key = jax.random.PRNGKey(seed)
    key, key_net = jax.random.split(key)
    key_env = jax.random.fold_in(key, 1)
    key_eval = jax.random.fold_in(key, 2)
    design_rng = np.random.default_rng(seed)

    jit_reset = jax.jit(
        lambda rngs, model: acting.reset(environment, rngs, model)
    )

    def sample_designs_and_model(rng, n):
        designs_np = model_lib.sample_designs(
            rng, n, design_low, design_high, design_dim
        )
        batched_model = model_lib.build_batched_model(generate_model_fn, designs_np)
        designs_input = model_lib.normalize_design(
            jnp.asarray(designs_np), design_low, design_high
        )
        return designs_np, batched_model, designs_input

    # Discover observation structure with a throwaway batch. MAICheetah obs is a dict
    # ({'state', 'privileged_state'}); keep the per-leaf trailing shape for the networks
    # and normalizer (brax selects the 'state' key internally).
    _, batched_model, _ = sample_designs_and_model(design_rng, num_envs)
    rngs = jax.random.split(key_env, num_envs)
    env_state = jit_reset(rngs, batched_model)
    obs_size = jax.tree_util.tree_map(lambda x: x.shape[-1], env_state.obs)

    # MAICheetah emits a multi-objective reward vector; collapse it to the single scalar
    # reward this algorithm optimizes via a fixed objective-weight vector (default ones).
    num_objectives = int(env_state.reward.shape[-1])
    if reward_objective_weights is None:
        reward_weights = jnp.ones(num_objectives)
    else:
        reward_weights = jnp.asarray(reward_objective_weights, dtype=jnp.float32)
    scalarize_reward = lambda r: jnp.sum(r * reward_weights, axis=-1)

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
    inference_fn = net_lib.make_design_inference_fn(design_networks)

    optimizer = optax.adam(learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
        )

    loss_fn = functools.partial(
        compute_design_hypernet_loss,
        design_networks=design_networks,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
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

    def training_step(carry, unused_t, batched_model, designs, first_state):
        training_state, state, key = carry
        key_sgd, key_unroll, new_key = jax.random.split(key, 3)
        policy = inference_fn(
            (training_state.normalizer_params, training_state.params.hypernetwork),
            designs,
        )

        def scan_unroll(c, _):
            cur_state, cur_key = c
            cur_key, nk = jax.random.split(cur_key)
            nstate, data = acting.generate_unroll(
                environment,
                cur_state,
                batched_model,
                policy,
                designs,
                cur_key,
                unroll_length,
                first_state,
                episode_length,
                scalarize_reward=scalarize_reward,
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
    def training_epoch(training_state, state, key, batched_model, designs, first_state):
        step = functools.partial(
            training_step,
            batched_model=batched_model,
            designs=designs,
            first_state=first_state,
        )
        (training_state, state, _), metrics = jax.lax.scan(
            step, (training_state, state, key), (), length=num_training_steps_per_epoch
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, state, metrics

    @jax.jit
    def eval_unroll(normalizer_params, hypernet_params, designs, batched_model, rngs, key):
        state = acting.reset(environment, rngs, batched_model)
        policy = inference_fn(
            (normalizer_params, hypernet_params), designs, deterministic=deterministic_eval
        )

        def body(carry, _):
            st, k, alive, ret = carry
            k, sub = jax.random.split(k)
            act, _ = policy(st.obs, sub)
            nst = jax.vmap(environment.step, in_axes=(0, 0, 0))(st, act, batched_model)
            ret = ret + scalarize_reward(nst.reward) * alive
            alive = alive * (1.0 - nst.done)
            return (nst, k, alive, ret), None

        init = (state, key, jnp.ones(num_eval_envs), jnp.zeros(num_eval_envs))
        (_, _, _, ret), _ = jax.lax.scan(body, init, (), length=episode_length)
        return ret

    def evaluate(training_state, key):
        _, eval_model, eval_designs = sample_designs_and_model(
            np.random.default_rng(int(key[0])), num_eval_envs
        )
        eval_rngs = jax.random.split(key, num_eval_envs)
        ret = eval_unroll(
            training_state.normalizer_params,
            training_state.params.hypernetwork,
            eval_designs,
            eval_model,
            eval_rngs,
            key,
        )
        ret = np.asarray(ret)
        return {
            "eval/episode_reward": float(np.mean(ret)),
            "eval/episode_reward_std": float(np.std(ret)),
        }

    # Initialize training state.
    init_params = DesignHypernetParams(
        hypernetwork=design_networks.hypernetwork.init(key_net)
    )
    normalizer_params = running_statistics.init_state(
        jax.tree_util.tree_map(
            lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
        )
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
        designs_np, batched_model, designs_input = sample_designs_and_model(
            design_rng, num_envs
        )
        key_env, sub = jax.random.split(key_env)
        rngs = jax.random.split(sub, num_envs)
        env_state = jit_reset(rngs, batched_model)
        first_state = env_state

        key, epoch_key = jax.random.split(key)
        t0 = time.time()
        training_state, env_state, train_metrics = training_epoch(
            training_state, env_state, epoch_key, batched_model, designs_input, first_state
        )
        train_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), train_metrics)
        epoch_time = time.time() - t0
        walltime += epoch_time
        current_step = (it + 1) * num_training_steps_per_epoch * env_step_per_training_step

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
