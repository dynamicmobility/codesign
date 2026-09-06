"""``design_lookup_hypernetwork``: networks and training algo.

A warm-up stage for the ``W`` of :mod:`design_hypernetwork`. ``M`` designs are drawn once by
Sobol and the feature map is hardcoded to a one-hot lookup over them, so the forward pass
collapses to ``flat(d_i) = W[i] + b``: row ``i`` of ``W`` holds design ``i``'s own policy and

    dL/dW[j] = sum_n features_j(d_n) * dL/dflat_n

is fed by design ``j``'s rollouts alone. After warm-up ``W`` is a basis of ``M`` trained
policies that a full run can start from via ``initialization_strategy: load_network``.

Two ``network_params`` knobs set where the ``M`` policies start and whether they stay
decoupled:

- ``initialization_strategy: experts`` (default) gives each row its own kaiming-initialized
  target network, so ``b + W[i]`` begins as an independently initialized policy.
  ``bias`` starts ``W`` at zero instead, so every design begins on the single kaiming policy
  held in ``b`` and the spread is learned.
- ``freeze_bias: true`` (default) holds ``b`` fixed, because ``dL/db`` sums over every design
  and would otherwise move all ``M`` policies on every step. Setting it false trains ``b``
  as a shared backbone, which is the point of pairing it with ``bias``.

The loss, inference fn and rollout machinery are ``design_hypernetwork``'s, unchanged.

Two things this does not give you:

- **Scale ``learning_rate`` down by ``sqrt(num_designs)``.** One design per minibatch means a
  row sees a gradient once every ``M`` steps, so Adam's second moment decays in between and
  the active-step update is ``eta * g / sqrt(v_hat) ~ eta * sqrt(M)``.
- The observation normalizer is shared across the grid, so the ``M`` policies are not
  literally ``M`` independent PPO runs.
"""

import functools
import warnings
from functools import partial
from typing import Callable, Literal, Sequence

import flax
import jax
import jax.numpy as jnp
import numpy as np
from brax.training import networks, types
from brax.training.acme import running_statistics
from brax.training.networks import Initializer
from flax import linen

from codesign.hyperdesigners import shared
from codesign.hyperdesigners.hypernetworks import (
    LookupA2CHypernet,
    flatten_model,
    sobol_design_table,
)
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    huber_loss,
    mse_loss,
)
from codesign.hyperdesigners.networks import (
    DesignHypernetNetworks,
    FeedForwardHypernetwork,
    make_target_networks,
)
from codesign.hyperdesigners.variants.design_hypernetwork import (
    compute_design_hypernet_loss,
    make_design_inference_fn,
)
from codesign.utils.grid import Grid


# --------------------------------------------------------------------------- networks
LookupInitStrategy = Literal["experts", "bias"]


def make_lookup_hypernetwork(
    obs_dim: int,
    target_policy_dict: dict,
    target_value_dict: dict,
    design_table: np.ndarray,
    policy_experts: jnp.ndarray | None = None,
    value_experts: jnp.ndarray | None = None,
    freeze_bias: bool = True,
    initialization_strategy: LookupInitStrategy = "experts",
) -> FeedForwardHypernetwork:
    """Wrap a ``LookupA2CHypernet`` over the ``M`` rows of ``design_table``.

    ``apply(params, design) -> (policy_params, value_params)``, as for the design
    hypernetwork.

    ``initialization_strategy`` picks where the ``M`` policies start:

    - ``experts``: ``policy_experts``/``value_experts`` are ``(M, num_params)`` flattened
      target-network draws and each row of ``W`` is that draw minus ``b``, so ``b + W[i]``
      is exactly the i-th draw and the anchors start independent.
    - ``bias``: ``W`` starts at zero, so every anchor starts on the single kaiming policy
      held in ``b`` and the per-design spread has to be learned into ``W``. This is
      ``design_hypernetwork``'s ``bias`` strategy, and pairs with ``freeze_bias=False``.
    """
    if initialization_strategy not in ("experts", "bias"):
        raise ValueError(
            f"Unsupported initialization_strategy: {initialization_strategy!r}"
        )
    if initialization_strategy == "experts" and (
        policy_experts is None or value_experts is None
    ):
        raise ValueError("the 'experts' strategy needs policy_experts and value_experts")
    num_designs = len(design_table)
    hypernet = LookupA2CHypernet(
        target_policy_dict=target_policy_dict,
        target_value_dict=target_value_dict,
        num_objs=design_table.shape[-1],
        obs_dim=obs_dim,
        num_features=num_designs,
        design_table=tuple(map(tuple, np.asarray(design_table, np.float32))),
        freeze_bias=freeze_bias,
    )
    dummy_design = jnp.asarray(design_table[0])

    def init(key):
        params = flax.core.unfreeze(hypernet.init(key, dummy_design))
        if initialization_strategy == "experts":
            params["params"]["policy_W"] = policy_experts - params["params"]["policy_b"]
            params["params"]["value_W"] = value_experts - params["params"]["value_b"]
        else:
            params["params"]["policy_W"] = jnp.zeros_like(params["params"]["policy_W"])
            params["params"]["value_W"] = jnp.zeros_like(params["params"]["value_W"])
        return flax.core.freeze(params)

    def apply(params, design):
        # Returns ((policy_params, value_params), (flat...), (features...)); take [0].
        return hypernet.apply(params, design)[0]

    def features(params, design):
        return hypernet.apply(params, design)[2][0]

    return FeedForwardHypernetwork(init=init, apply=apply, features=features)


def make_lookup_hypernet_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    design_dim: int,
    key: jax.Array,
    num_designs: int = 8,
    design_seed: int = 0,
    design_low: Sequence[float] = (0.0,),
    design_high: Sequence[float] = (1.0,),
    freeze_bias: bool = True,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (64,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (64,) * 2,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    num_value_outputs: int = 1,
    weight_initializer: str | Initializer = "kaiming_uniform",
    initialization_strategy: LookupInitStrategy = "experts",
) -> DesignHypernetNetworks:
    """Build the target MLPs and a lookup hypernetwork over ``num_designs`` fixed designs.

    Every keyword is JSON-serializable so that brax's checkpoint config can carry it and
    :mod:`codesign.learning.inference` can rebuild the design table without a live env.
    """
    if design_dim != len(design_low):
        raise ValueError(
            f"design_dim {design_dim} does not match len(design_low) {len(design_low)}"
        )

    (
        parametric_action_distribution,
        policy_network,
        value_network,
        target_policy_params,
        target_value_params,
        obs_dim,
    ) = make_target_networks(
        observation_size=observation_size,
        action_size=action_size,
        key=key,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
        num_value_outputs=num_value_outputs,
        weight_initializer=weight_initializer,
    )

    # One independent draw of the target nets per design, so each row of W starts as its own
    # network rather than as a flat perturbation of a shared one. 'bias' starts every design
    # on the shared policy in b instead, so the draws are not needed.
    policy_experts = value_experts = None
    if initialization_strategy == "experts":
        flatten = lambda p: flatten_model(p["params"])[0]
        expert_keys = jax.random.split(jax.random.fold_in(key, 1), num_designs)
        policy_experts = jnp.stack([flatten(policy_network.init(k)) for k in expert_keys])
        value_experts = jnp.stack([flatten(value_network.init(k)) for k in expert_keys])

    hypernetwork = make_lookup_hypernetwork(
        obs_dim=obs_dim,
        target_policy_dict=target_policy_params,
        target_value_dict=target_value_params,
        design_table=sobol_design_table(design_seed, num_designs, design_low, design_high),
        policy_experts=policy_experts,
        value_experts=value_experts,
        freeze_bias=freeze_bias,
        initialization_strategy=initialization_strategy,
    )

    return DesignHypernetNetworks(
        hypernetwork=hypernetwork,
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )



# ------------------------------------------------------------------------ evaluation
def warn_off_table(design, table) -> None:
    """Warn when ``design`` sits farther from every anchor than the anchors sit apart.

    The lookup snaps any design to its nearest anchor, so an off-table design quietly
    returns a neighbouring design's policy driving a robot that policy never trained on.
    ``design`` is normalized, matching ``table``; a traced design is skipped, there being
    nothing concrete to measure.
    """
    try:
        queried = np.atleast_2d(np.asarray(design, np.float32))
    except jax.errors.TracerArrayConversionError:
        return

    table = np.asarray(table, np.float32)
    gaps = np.linalg.norm(table[:, None] - table[None], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    spacing = float(np.median(gaps.min(axis=1)))

    distance = np.linalg.norm(queried[:, None] - table[None], axis=-1)
    nearest, offset = distance.argmin(axis=1), distance.min(axis=1)
    for i in np.flatnonzero(offset > spacing):
        warnings.warn(
            f"design {queried[i]} is {offset[i]:.3g} from its nearest anchor (index "
            f"{nearest[i]}, typical anchor spacing {spacing:.3g}); the lookup returns "
            "that anchor's policy, which was not trained on this design",
            stacklevel=3,
        )


# -------------------------------------------------------------------- training
def train_design_lookup_hypernetwork(
    environment,
    num_timesteps: int,
    episode_length: int,
    num_parallel_envs: int = 128,
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
    design_low: Sequence[float] = (0.0,),
    design_high: Sequence[float] = (1.0,),
    num_designs: int = 8,
    network_factory: Callable = make_lookup_hypernet_networks,
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
    assert num_parallel_envs % num_designs == 0, (
        "num_parallel_envs must be divisible by num_designs"
    )
    assert num_eval_envs % num_designs == 0, (
        "num_eval_envs must be divisible by num_designs"
    )
    # One design per minibatch, so a row of W is updated by that design's rollouts alone.
    assert num_minibatches == num_designs, (
        "num_minibatches must equal num_designs for one design per minibatch"
    )
    per_cell = num_parallel_envs // num_designs
    eval_per_cell = num_eval_envs // num_designs
    # The design set is fixed, so there is nothing to resample.
    schedule = shared.Schedule.make(
        num_timesteps, num_evals, num_parallel_envs, batch_size, num_minibatches, unroll_length, 1,
    )

    key = jax.random.PRNGKey(seed)
    key, key_net = jax.random.split(key)
    limits = (np.asarray(design_low, np.float32), np.asarray(design_high, np.float32))

    normalize = (
        running_statistics.normalize if normalize_observations else (lambda x, y: x)
    )
    design_networks = network_factory(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        design_dim=design_dim,
        key=key_net,
        preprocess_observations_fn=normalize,
        num_designs=num_designs,
        design_seed=seed,
        design_low=tuple(design_low),
        design_high=tuple(design_high),
    )
    inference_fn = make_design_inference_fn(design_networks)
    # This hypernetwork is keyed on the design alone, so the trivial tradeoff the grid
    # carries goes unused.
    make_policy = lambda norm, params, designs, tradeoffs, **kw: inference_fn(
        (norm, params.hypernetwork), designs, **kw
    )

    optimizer = shared.make_optimizer(learning_rate, max_grad_norm)
    loss_fn = functools.partial(
        compute_design_hypernet_loss,
        design_networks       = design_networks,
        entropy_cost          = entropy_cost,
        discounting           = discounting,
        reward_scaling        = reward_scaling,
        gae_lambda            = gae_lambda,
        clipping_epsilon      = clipping_epsilon,
        normalize_advantage   = normalize_advantage,
        value_loss_fn         = partial(huber_loss, huber_delta = huber_delta) if value_loss_type == 'huber' else mse_loss,
    )
    chunk = shared.make_training_chunk(
        environment, make_policy,
        shared.make_sgd_step(loss_fn, optimizer, num_minibatches, 'design'),
        schedule, unroll_length, episode_length, num_updates_per_batch,
    )
    rollout_returns = shared.make_rollout_returns(
        environment, make_policy, episode_length, deterministic_eval
    )

    # An int seed rebuilds the same Sobol draw every call, so training and eval run on the
    # same M designs -- an eval design off the table would snap to the wrong row.
    build_inputs = shared.make_env_inputs(environment)
    eval_grid = Grid.from_design_sample(
        environment, seed, num_designs, per_cell=eval_per_cell, limits=limits
    )
    eval_model, eval_designs, eval_tradeoffs = build_inputs(eval_grid)
    train_grid = Grid.from_design_sample(
        environment, seed, num_designs, per_cell=per_cell, limits=limits
    )
    train_inputs = build_inputs(train_grid)

    # The designs the envs run on must be the table the hypernetwork looks up; this catches
    # ``env.design_limits`` disagreeing with the configured box.
    table = sobol_design_table(seed, num_designs, design_low, design_high)
    for designs, stride in ((train_inputs[1], per_cell), (eval_designs, eval_per_cell)):
        np.testing.assert_allclose(np.asarray(designs)[::stride], table, atol=1e-6)

    # Sampled once; the M models are built once rather than once per epoch.
    sample = lambda it, extra_state, key: (train_grid, None)
    env_inputs = lambda grid: train_inputs

    def evaluate(training_state, extra_state, key):
        rewards = rollout_returns(
            training_state.normalizer_params,
            training_state.params,
            eval_designs,
            eval_tradeoffs,
            eval_model,
            shared.paired_eval_keys(key, num_designs, eval_per_cell),
            key,
        )
        metrics = shared.eval_metrics(jnp.sum(rewards, axis=0), eval_grid)
        return {**metrics, **shared.per_design_metrics(metrics["eval_grid"])}

    params_of = lambda ts, extra: (ts.normalizer_params, ts.params.hypernetwork)

    training_state = shared.init_training_state(
        DesignHypernetParams(hypernetwork=design_networks.hypernetwork.init(key_net)),
        optimizer, environment.observation_size,
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
        key,
        inference_fn,
        env_inputs,
        run_evals=run_evals,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    return inference_fn, params, metrics


# ----------------------------------------------------------------------- setup
def setup_design_lookup_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the lookup (warm-up) hypernetwork."""
    lp = config["learning_params"]
    ppo = dict(lp["ppo_params"])
    net = dict(lp["network_params"])
    design = dict(config['env_config']['codesign'])
    design_sampling = dict(lp.get("design_sampling", {}))

    # The design table defines the network, so it is bound here and travels in the
    # checkpoint config; ``train_fn`` passes the same values again at call time.
    network_factory = functools.partial(
        make_lookup_hypernet_networks,
        num_designs                 = design_sampling.get("num_designs", 8),
        design_seed                 = ppo["seed"],
        design_low                  = tuple(design["low"]),
        design_high                 = tuple(design["high"]),
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
        weight_initializer          = net.get("weight_initializer", "kaiming_uniform"),
        freeze_bias                 = net.get("freeze_bias", True),
        initialization_strategy     = net.get("initialization_strategy", "experts"),
    )

    train_fn = functools.partial(
        train_design_lookup_hypernetwork,
        network_factory     = network_factory,
        design_dim          = len(design['low']),
        design_low          = tuple(design['low']),
        design_high         = tuple(design['high']),
        num_designs         = design_sampling.get("num_designs", 8),
        **ppo,
    )
    return train_fn, network_factory
