"""Training scaffolding shared by the ``hyperdesigners`` algos.

Each algo brings its own networks, loss, and sampling strategy; the PPO machinery around
them -- schedule arithmetic, minibatched SGD, the unroll, the eval rollout, and the epoch
loop -- is the same, and lives here.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from typing import Any, Callable, NamedTuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import gradients
from brax.training.acme import running_statistics

from codesign.hyperdesigners import acting
from codesign.utils import model as model_lib
from codesign.utils.grid import Grid, DesignTransition


@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: Any  # the algo's trainable params, e.g. DesignHypernetParams
    normalizer_params: running_statistics.RunningStatisticsState


class TrainingBatch(NamedTuple):
    """Step transitions plus the per-environment conditioning owned by their grid."""

    transitions: acting.DesignTransition
    designs: jax.Array
    tradeoffs: jax.Array


@dataclasses.dataclass
class Schedule:
    """The step budget, split into epochs (one eval each), resample chunks, and steps.
    """

    num_scans                       : int  # number of scans which eventually concatenate into one training dataset
    env_step_per_training_step      : int
    num_training_steps_per_resample : int
    num_epochs                      : int
    resamples_per_epoch             : int

    @classmethod
    def make(
        cls,
        num_timesteps: int,
        num_evals: int,
        num_envs: int,
        batch_size: int,
        num_minibatches: int,
        unroll_length: int,
        resamples_per_epoch: int = 1,
    ) -> Schedule:
        assert (batch_size * num_minibatches) % num_envs == 0, (
            "batch_size * num_minibatches must be divisible by num_envs"
        )
        assert resamples_per_epoch >= 1, "resamples_per_epoch must be >= 1"
        env_step_per_training_step = batch_size * unroll_length * num_minibatches
        num_epochs = max(num_evals - 1, 1)
        return cls(
            num_scans                    = batch_size * num_minibatches // num_envs,
            env_step_per_training_step   = env_step_per_training_step,
            num_training_steps_per_resample = int(np.ceil(
                num_timesteps
                / (num_epochs * resamples_per_epoch * env_step_per_training_step)
            )),
            num_epochs                   = num_epochs,
            resamples_per_epoch          = resamples_per_epoch,
        )

    @property
    def num_training_steps_per_epoch(self) -> int:
        return self.num_training_steps_per_resample * self.resamples_per_epoch

    @property
    def env_step_per_epoch(self) -> int:
        return self.num_training_steps_per_epoch * self.env_step_per_training_step


def make_optimizer(learning_rate: float, max_grad_norm: float | None = None):
    """Adam, preceded by global-norm gradient clipping when ``max_grad_norm`` is set."""
    if max_grad_norm is None:
        return optax.adam(learning_rate)
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
    )


def init_training_state(params, optimizer, observation_size) -> TrainingState:
    """Optimizer state and observation normalizer for a fresh set of ``params``.

    ``observation_size`` is what the normalizer covers, which is the env's own size unless
    the algo widens the observation (design_mlp appends the design to it).
    """
    return TrainingState(
        optimizer_state=optimizer.init(params),
        params=params,
        normalizer_params=running_statistics.init_state(
            model_lib.observation_spec(observation_size)
        ),
    )


def make_env_inputs(env) -> Callable:
    """``grid -> (batched_model, designs, tradeoffs)`` on the flat env axis.

    The first stacked model's treedef is pinned onto later ones so that resampling the
    grid does not retrace the jitted training and rollout functions.
    """
    reference = None

    def env_inputs(grid: Grid):
        nonlocal reference
        batched_model, designs, tradeoffs = grid.env_inputs(env, like=reference)
        reference = batched_model
        return batched_model, designs, tradeoffs

    return env_inputs


def make_sgd_step(loss_fn, optimizer, num_minibatches: int, batching_strategy = 'design') -> Callable:
    """``sgd_step(carry, _, data, normalizer_params)``: one shuffled pass of minibatched
    gradient steps over ``data``, carrying ``(optimizer_state, params, key)``.
    If batching_strategy is ``design`` then num_minibatches should be equal to grid num designs
    """
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=None, has_aux=True
    )

    def flatten_transitions(x):
        return x.reshape((-1,) + x.shape[3:])

    def get_transitions_from_grid(grid: Grid) -> DesignTransition:
        return jax.tree_util.tree_map(functools.partial(flatten_transitions), grid.transitions)

    def shuffle(g: Grid, num_minibatches, key):
        def convert(x):
            x = jax.random.permutation(key, x)
            return jnp.swapaxes(jnp.reshape(x, (num_minibatches, -1) + x.shape[1:]), 1, 2)
        transitions = get_transitions_from_grid(g)
        print("Transitions before Shuffle: ",transitions.reward.shape)
        data_shuffled = jax.tree_util.tree_map(functools.partial(convert), transitions)
        return data_shuffled

    # Pretty sure this doesn't work
    def batch_design(x:Grid, num_minibatches, key) -> DesignTransition:
        design_grids = x.batch_by_design(key, num_minibatches)
        def convert(x):
            return jnp.swapaxes(x, 1, 2)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[get_transitions_from_grid(grid) for grid in design_grids])
        shuffled_stacked = jax.tree_util.tree_map(functools.partial(convert), stacked)
        return shuffled_stacked

    batch_fn = batch_design if batching_strategy == 'design' else shuffle

    def batch_step(carry, data: DesignTransition, normalizer_params):
        opt_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, opt_state = gradient_update_fn(
            params, normalizer_params, data, key_loss, optimizer_state=opt_state
        )
        return (opt_state, params, key), metrics

    def sgd_step(carry, unused_t, data: Grid, normalizer_params):
        opt_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        # The batching function should return a list of collections of transitions whose leading dimension is the 
        # shuffled = jax.tree_util.tree_map(functools.partial(shuffle, num_minibatches=num_minibatches, key=key_perm), data)
        shuffled = batch_fn(data, num_minibatches, key_perm)
        print("Transitions after Shuffle: ", shuffled.reward.shape)
        (opt_state, params, _), metrics = jax.lax.scan(
            functools.partial(batch_step, normalizer_params=normalizer_params),
            (opt_state, params, key_grad),
            shuffled,
            length=num_minibatches,
        )
        return (opt_state, params, key), metrics

    return sgd_step


def make_training_chunk(
    env,
    make_policy: Callable,
    sgd_step: Callable,
    schedule: Schedule,
    unroll_length: int,
    episode_length: int,
    num_updates_per_batch: int,
    observation_fn: Callable | None = None,
) -> Callable:
    """Jitted training against one fixed design x tradeoff grid.

    Returns ``training_chunk(training_state, state, key, batched_model, designs, tradeoffs,
    first_state) -> (training_state, state, metrics)``, which collects rollouts under the
    current networks, refreshes the observation normalizer, and takes PPO steps,
    ``num_training_steps_per_chunk`` times over. ``observation_fn`` pulls the observation
    the normalizer covers out of a batch of transitions.
    """
    observation_fn = observation_fn or (lambda grid: grid.transitions.observation)

    def training_step(carry, unused_t, batched_model, designs, tradeoffs, grid, first_state):
        training_state, state, key = carry
        key_sgd, key_unroll, new_key = jax.random.split(key, 3)
        policy = make_policy(
            training_state.normalizer_params,
            training_state.params,
            designs,
            tradeoffs,
        )

        def scan_unroll(c, _):
            cur_state, cur_key = c
            cur_key, nk = jax.random.split(cur_key)
            nstate, data = acting.generate_unroll(
                env,
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
        (state, _), transitions = jax.lax.scan(
            scan_unroll, (state, key_unroll), (), length=schedule.num_scans
        )
        # Instead of putting the transitions into a TrainingBatch, reshape the transition data and put it into a Grid.
        # Make sure the leading 3 dimensions (M, K, C) correspond to the designs, tradeoffs, and per_cell properties of the grid.
        # This information is given by the argument grid_shape
        m, k, c = grid.cell_shape

        def reshape_transitions(x):
            # [S, T, M*K*C, ...] -> [M, K, C*S, T, ...].  Keeping C and S
            # adjacent before merging them preserves every transition's grid cell.
            x = x.reshape(
                (schedule.num_scans, x.shape[1], m, k, c) + x.shape[3:]
            )
            x = jnp.transpose(x, (2, 3, 4, 0, 1) + tuple(range(5, x.ndim)))
            return x.reshape(
                (m, k, c * schedule.num_scans, x.shape[4]) + x.shape[5:]
            )

        data = Grid(
            designs     = grid.designs,
            tradeoffs   = grid.tradeoffs,
            per_cell    = c * schedule.num_scans,
            transitions = jax.tree_util.tree_map(reshape_transitions, transitions),
        )

        # normalize observations based on current state distribution
        normalizer_params = running_statistics.update(
            training_state.normalizer_params, observation_fn(data)
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
        training_state, state, key, batched_model, designs, tradeoffs, grid, first_state
    ):
        step = functools.partial(
            training_step,
            batched_model=batched_model,
            designs=designs,
            tradeoffs=tradeoffs,
            first_state=first_state,
            grid = grid
        )
        # Does ``num_training_steps_per_resample`` rollouts -> sgd backprops
        (training_state, state, _), metrics = jax.lax.scan(
            step,
            (training_state, state, key),
            (),
            length=schedule.num_training_steps_per_resample,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, state, metrics

    return training_chunk


def make_rollout_returns(
    env, make_policy: Callable, episode_length: int, deterministic: bool = True
) -> Callable:
    """Jitted per-step reward of one episode per env, zeroed after termination.

    Returns ``rollout_returns(normalizer_params, params, designs, tradeoffs, batched_model,
    rngs, key) -> [episode_length, num_envs, *reward_shape]``. An env that terminates early
    contributes zero from its termination step onward.
    """

    @jax.jit
    def rollout_returns(
        normalizer_params, params, designs, tradeoffs, batched_model, rngs, key
    ):
        state = acting.reset(env, rngs, batched_model)
        policy = make_policy(
            normalizer_params,
            params,
            designs,
            tradeoffs,
            deterministic=deterministic,
        )

        def body(carry, _):
            st, k, alive = carry
            k, sub = jax.random.split(k)
            act, _ = policy(st.obs, sub)
            nst = jax.vmap(env.step, in_axes=(0, 0, 0))(st, act, batched_model)
            # Broadcast alive [num_envs] over a scalar or per-objective reward.
            alive_b = alive.reshape(alive.shape + (1,) * (nst.reward.ndim - alive.ndim))
            reward = nst.reward * alive_b
            alive = alive * (1.0 - nst.done)
            return (nst, k, alive), reward

        init = (state, key, jnp.ones(rngs.shape[0]))
        _, rewards = jax.lax.scan(body, init, (), length=episode_length)
        return rewards

    return rollout_returns


def eval_metrics(returns, grid) -> dict:
    """Eval metrics for a rolled-out grid.

    ``returns`` is the episode return on the flat env axis, ``[num_envs]`` or
    ``[num_envs, n_r]``. Reports the tradeoff-scalarized reward and its spread, a mean per
    objective, and the grid itself (per-objective returns per cell) for Pareto plotting.
    """
    returns = np.asarray(returns).reshape(grid.num_envs, grid.n_r)
    _, tradeoffs = grid.flatten()
    scalarized = np.sum(tradeoffs * returns, axis=-1)
    metrics = {
        "eval/episode_reward": float(np.mean(scalarized)),
        "eval/episode_reward_std": float(np.std(scalarized)),
    }
    for i in range(grid.n_r):
        metrics[f"eval/episode_reward_obj{i}"] = float(np.mean(returns[:, i]))
    metrics["eval_grid"] = dataclasses.replace(grid, rewards=grid.unflatten(returns))
    return metrics


class Sampled(NamedTuple):
    """One resample chunk's grid and the rollout inputs built from it."""

    grid      : Any
    model     : Any
    designs   : jax.Array
    tradeoffs : jax.Array
    aux       : Any


@dataclasses.dataclass
class Algorithm:
    """The pieces of a training run that differ between algos.

    - ``sample(it, extra_state, key) -> (Grid, aux)`` draws the chunk's design x tradeoff
    grid; ``aux`` is passed on to ``post_chunk`` and otherwise unused. 
    
    - ``chunk`` comes from :func:`make_training_chunk`. 
    
    - ``evaluate(training_state, extra_state, key)``
    
    - ``params_of(training_state, extra_state)`` produces the epoch's metrics and its
    checkpointed params. 
    
    - ``post_chunk(training_state, extra_state, sampled, key) -> (extra_state, metrics)`` 
    runs after every chunk, for algos training a second model.
    
    - ``epoch_metrics(it)`` adds per-epoch bookkeeping.
    """

    sample        : Callable[..., tuple[Grid, dict]] # sampling strategy fn
    chunk         : Callable # training chunk fn
    evaluate      : Callable # evaluation fn
    params_of     : Callable # get network-specific params
    post_chunk    : Callable = lambda ts, extra, sampled, key: (extra, {}) # runs after every chunk
    epoch_metrics : Callable = lambda it: {} # per-epoch metrics


def run_training(
    algo: Algorithm,
    schedule: Schedule,
    training_state: TrainingState,
    env,
    key: jax.Array,
    inference_fn: Callable,
    env_inputs: Callable,
    extra_state=None,
    run_evals: bool = True,
    progress_fn: Callable = lambda *a: None,
    policy_params_fn: Callable = lambda *a: None,
):
    """Resample, train, evaluate, checkpoint -- once per epoch.

    Returns ``(params, metrics)``: the last checkpointed params and the last epoch's metrics.
    """
    jit_reset = jax.jit(lambda rngs, model: acting.reset(env, rngs, model))
    key_env = jax.random.fold_in(key, 1)
    key_eval = jax.random.fold_in(key, 2)
    key_sample = jax.random.fold_in(key, 3)

    # Initial eval + checkpoint.
    metrics = {}
    metrics = algo.evaluate(training_state, extra_state, key_eval)
    progress_fn(0, metrics)
    params = algo.params_of(training_state, extra_state)
    policy_params_fn(0, inference_fn, params)

    walltime = 0.0
    for it in range(schedule.num_epochs):
        t0 = time.time()
        chunk_metrics = []
        for _ in range(schedule.resamples_per_epoch):
            # Redraw the grid, rebuild the per-env models, and restart the envs on them
            # (the robot itself changed, so the carried state is stale).
            # TODO: sub is generateed from jax split but algo.sample will use a numpy rng (as it is on the CPU side). 
            # The sampler should be written to be consistent with that
            key_sample, sub = jax.random.split(key_sample)
            grid, aux = algo.sample(it, extra_state, sub)
            # Sampled holds flattened model, designs, tradeoffs, but are consistent with the grid shape
            sampled = Sampled(grid, *env_inputs(grid), aux)

            key_env, sub = jax.random.split(key_env)
            env_state = jit_reset(jax.random.split(sub, grid.num_envs), sampled.model)

            key, chunk_key = jax.random.split(key)
            training_state, _, train_metrics = algo.chunk(
                training_state, env_state, chunk_key, sampled.model,
                sampled.designs, sampled.tradeoffs, grid, env_state,
            )

            key_env, sub = jax.random.split(key_env)
            extra_state, post_metrics = algo.post_chunk(
                training_state, extra_state, sampled, sub
            )
            chunk_metrics.append({**train_metrics, **post_metrics})

        train_metrics = jax.tree_util.tree_map(
            lambda *xs: jnp.mean(jnp.stack(xs)), *chunk_metrics
        )
        train_metrics = jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), train_metrics
        )
        epoch_time = time.time() - t0
        walltime += epoch_time
        current_step = (it + 1) * schedule.env_step_per_epoch

        metrics = {
            "training/sps": schedule.env_step_per_epoch / epoch_time,
            "training/walltime": walltime,
            **algo.epoch_metrics(it),
            **{f"training/{k}": float(v) for k, v in train_metrics.items()},
        }
        if run_evals:
            key_eval, eval_subkey = jax.random.split(key_eval)
            metrics.update(algo.evaluate(training_state, extra_state, eval_subkey))

        params = algo.params_of(training_state, extra_state)
        policy_params_fn(current_step, inference_fn, params)
        progress_fn(current_step, metrics)

    return params, metrics
