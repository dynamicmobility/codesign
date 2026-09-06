"""``design_mlp``: networks, loss and training algo.

A plain policy/value MLP pair conditioned on the design by appending it to the
observation, rather than by a hypernetwork generating the weights from it.
"""

import functools
from functools import partial
from typing import Any, Callable, Literal, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.types import Params, PRNGKey
from flax import linen

from codesign.hyperdesigners.networks import make_vector_value_network
from codesign.hyperdesigners import shared
from codesign.hyperdesigners.losses import huber_loss, mse_loss
from codesign.utils.grid import DesignTransition, Grid


# --------------------------------------------------------------------------- networks
# Represents an MLP conditioned on design for single-objective multi-design
@flax.struct.dataclass
class DesignNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def augment_observation_size(
    observation_size: types.ObservationSize, design_dim: int
) -> types.ObservationSize:
    """Add the appended design width to every observation leaf's feature axis."""
    return jax.tree_util.tree_map(
        lambda size: (
            size[:-1] + (size[-1] + design_dim,)
            if isinstance(size, tuple)
            else size + design_dim
        ),
        observation_size,
        is_leaf=lambda size: isinstance(size, tuple),
    )


def make_design_mlp_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    design_dim: int,
    key: jax.Array,
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
)->DesignNetworks:
    """Build the target policy/value MLPs and the design-conditioned hypernetwork."""
    if distribution_type == "normal":
        parametric_action_distribution = distribution.NormalDistribution(
            event_size=action_size
        )
    elif distribution_type == "tanh_normal":
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
    else:
        raise ValueError(
            f'Unsupported distribution type: {distribution_type}. Must be one'
            ' of "normal" or "tanh_normal".'
        )
    policy_network = networks.make_policy_network(
        param_size=parametric_action_distribution.param_size,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
    )

    value_network = make_vector_value_network(
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        num_objectives=num_value_outputs,
        activation=activation,
        obs_key=value_obs_key,
    )

    return DesignNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

def make_design_mlp_inference_fn(networks_: DesignNetworks):
    """Inference-fn factory keyed on the robot design.

    Returns ``inference_fn(params, design, deterministic=False) -> policy(obs, key)``,
    where ``params = (normalizer_params, hypernet_params)``. ``design`` may be a single
    design ``(design_dim,)`` or batched ``(num_envs, design_dim)``; in the batched case
    obs/params are vmapped over the leading env axis.
    """

    def design_mlp_inference_fn(
        params: types.Params, design: jax.Array, deterministic: bool = False
    ) -> types.Policy:
        normalizer_params, policy_params = params[0], params[1]
        policy_network = networks_.policy_network
        parametric_action_distribution = networks_.parametric_action_distribution

        if len(design.shape) == 1:
            policy_apply = policy_network.apply
        else:
            policy_apply = jax.vmap(policy_network.apply, in_axes=(None, None, 0))

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            design_input = jax.tree_util.tree_map(
                lambda obs: jnp.broadcast_to(
                    design, obs.shape[:-1] + design.shape[-1:]
                ),
                observations,
            )

            logits = policy_apply(
                normalizer_params,
                policy_params,
                jax.tree_util.tree_map(
                    lambda obs, des: jnp.concatenate((obs, des), axis=-1),
                    observations,
                    design_input,
                ),
            )
            if deterministic:
                return parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return design_mlp_inference_fn


# ------------------------------------------------------------------------ loss
@flax.struct.dataclass
class DesignMLPParams:
    """Trainable parameters: the design-conditioned policy and value MLPs."""

    policy_params: Params
    value_params: Params

def compute_design_mlp_loss(
    params: DesignMLPParams,
    normalizer_params: Any,
    data: DesignTransition,
    rng: jnp.ndarray,
    design_networks: DesignNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    value_loss_fn: Callable = mse_loss,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes the clipped-PPO loss for the design hypernetwork.

    Args:
        params: trainable hypernetwork params.
        normalizer_params: observation normalizer params.
        data: ``DesignTransition`` with leading dimensions ``[B, T]``. Its ``design``
            field contains the normalized per-environment design repeated over time and
            requires
            ``extras['state_extras']['truncation']``,
            ``extras['policy_extras']['raw_action']``, and
            ``extras['policy_extras']['log_prob']``.
        rng: PRNG key (for entropy estimate).
        design_networks: the design hypernetwork bundle.
    """
    parametric_action_distribution = design_networks.parametric_action_distribution
    policy_apply = jax.vmap(
        design_networks.policy_network.apply, in_axes=(None, None, 1)
    )
    value_apply = jax.vmap(
        design_networks.value_network.apply, in_axes=(None, None, 1)
    )
    single_value_apply = jax.vmap(
        design_networks.value_network.apply, in_axes=(None, None, 0)
    )

    policy_params = params.policy_params
    value_params = params.value_params

    # Put the time dimension first: [B, T, ...] -> [T, B, ...].
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    observation = jax.tree_util.tree_map(
        lambda obs: jnp.concatenate((obs, data.design), axis=-1),
        data.observation,
    )
    policy_logits = policy_apply(normalizer_params, policy_params, observation)
    policy_logits = jnp.swapaxes(policy_logits, 0, 1)

    baseline = value_apply(normalizer_params, value_params, observation)
    baseline = jnp.swapaxes(baseline, 0, 1)

    terminal_obs = jax.tree_util.tree_map(
        lambda obs: jnp.concatenate((obs[-1], data.design[-1]), axis=-1),
        data.next_observation,
    )
    bootstrap_value = single_value_apply(
        normalizer_params, value_params, terminal_obs
    )

    rewards = data.reward * reward_scaling  # [T, B] (single objective)

    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = ppo_losses.compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss.
    v_error = vs - baseline
    v_loss = value_loss_fn(v_error)

    # Entropy bonus.
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    total_loss = policy_loss + v_loss + entropy_loss

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
    }


# -------------------------------------------------------------------- training
DESIGN_STRATEGIES = ("random", "fixed", "noisy-fixed")


def train_design_mlp(
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
    num_designs: int = 8,
    resamples_per_epoch: int = 1,
    strategy: str = "random",
    warmup_strategy: str | None = None,
    warmup_epochs: int = 0,
    rate: float = 0.0,
    network_factory: Callable = make_design_mlp_networks,
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
    warmup_strategy = strategy if warmup_strategy is None else warmup_strategy
    for name, value in (("strategy", strategy), ("warmup_strategy", warmup_strategy)):
        if value not in DESIGN_STRATEGIES:
            raise ValueError(
                f"Unsupported {name}: {value!r}; expected one of {DESIGN_STRATEGIES}"
            )
    if strategy == "fixed" and resamples_per_epoch != 1:
        raise ValueError(
            "strategy 'fixed' redraws nothing, so resamples_per_epoch must be 1"
        )
    if "noisy-fixed" in (strategy, warmup_strategy) and rate <= 0.0:
        raise ValueError(
            f"'noisy-fixed' needs rate > 0, got {rate}; at rate 0 the std never "
            "leaves zero and the strategy is just 'fixed'"
        )
    if warmup_strategy != strategy and warmup_epochs <= 0:
        raise ValueError(
            f"warmup_strategy {warmup_strategy!r} differs from strategy {strategy!r} "
            f"but warmup_epochs is {warmup_epochs}, so the warm-up never runs"
        )
    schedule = shared.Schedule.make(
        num_timesteps, num_evals, num_parallel_envs, batch_size, num_minibatches,
        unroll_length, resamples_per_epoch,
    )
    if warmup_strategy != strategy and warmup_epochs >= schedule.num_epochs:
        raise ValueError(
            f"warmup_epochs {warmup_epochs} covers all {schedule.num_epochs} epochs "
            f"(num_evals - 1), so strategy {strategy!r} never runs"
        )

    key = jax.random.PRNGKey(seed)
    key, key_net = jax.random.split(key)
    design_rng = np.random.default_rng(seed)

    # The design rides along in the observation, so every observation the networks and the
    # normalizer see is design_dim wider than the env's own.
    obs_size = augment_observation_size(environment.observation_size, design_dim)

    def append_design(data: Grid):
        return jax.tree_util.tree_map(
            lambda obs: jnp.concatenate((obs, data.transitions.design), axis=-1),
            data.transitions.observation,
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
    inference_fn = make_design_mlp_inference_fn(design_networks)
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
        value_loss_fn         = partial(huber_loss, huber_delta = huber_delta) if value_loss_type == 'huber' else mse_loss,
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

    per_cell = num_parallel_envs // num_designs
    anchor_grid = Grid.from_design_sample(
        environment, seed, num_designs, per_cell=per_cell
    )
    anchors = anchor_grid.designs[:, 0]  # (num_designs, design_dim), physical units
    design_low, design_high = np.asarray(environment.design_limits, np.float32)
    design_span = design_high - design_low

    def draw(name: str, iteration: int) -> Grid:
        """Draw one design grid for the named strategy."""
        if name == "fixed":
            return anchor_grid
        if name == "random":
            return Grid.from_design_sample(
                environment, design_rng, num_designs, per_cell=per_cell
            )
        noise = design_rng.normal(0.0, 1.0, anchors.shape) * (
            rate * iteration * design_span
        )
        return Grid.crossed(
            np.clip(anchors + noise, design_low, design_high),
            np.ones((1, 1), np.float32),
            per_cell,
        )

    phase, iteration = None, 0

    def sample(it, extra_state, key):
        """Sample designs using the warm-up or main strategy for this epoch."""
        nonlocal phase, iteration
        current = warmup_strategy if it < warmup_epochs else strategy
        if current != phase:
            phase, iteration = current, 0
        grid = draw(current, iteration)
        iteration += 1
        return grid, None

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
        key,
        inference_fn,
        env_inputs,
        run_evals=run_evals,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    return inference_fn, params, metrics


# ----------------------------------------------------------------------- setup
def setup_design_mlp(config):
    """Return ``(train_fn, network_factory)`` for the design-conditioned MLP.
    """
    lp                = config["learning_params"]
    ppo               = dict(lp["ppo_params"])
    net               = dict(lp["network_params"])
    codesign          = dict(config['env_config']['codesign'])
    design_sampling   = dict(lp.get('design_sampling', {}))

    if "sampling" in design_sampling:
        raise ValueError(
            "design_sampling 'sampling' is now 'strategy', optionally preceded by a "
            "'warmup_strategy' for the first 'warmup_epochs' epochs"
        )

    network_factory = functools.partial(
        make_design_mlp_networks,
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
    )

    train_fn = functools.partial(
        train_design_mlp,
        network_factory       = network_factory,
        design_dim            = len(codesign["low"]),
        num_designs           = design_sampling.get("num_designs", 8),
        resamples_per_epoch   = design_sampling.get("resamples_per_epoch", 1),
        strategy              = design_sampling.get("strategy", "random"),
        warmup_strategy       = design_sampling.get("warmup_strategy"),
        warmup_epochs         = design_sampling.get("warmup_epochs", 0),
        rate                  = design_sampling.get("rate", 0.0),
        **ppo,
    )
    return train_fn, network_factory
