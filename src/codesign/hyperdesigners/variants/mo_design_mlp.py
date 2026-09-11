"""``mo_design_mlp``: networks, loss and training algo.

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
from codesign.utils.grid import DesignTransition, Grid, sample_tradeoffs_cpu


# --------------------------------------------------------------------------- networks
# Represents an MLP conditioned on design for single-objective multi-design
@flax.struct.dataclass
class DesignNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def augment_observation_size(
    observation_size: types.ObservationSize, design_dim: int, num_objectives: int
) -> types.ObservationSize:
    """Add design and tradeoff widths to every observation leaf's feature axis."""
    return jax.tree_util.tree_map(
        lambda size: (
            size[:-1] + (size[-1] + design_dim + num_objectives,)
            if isinstance(size, tuple)
            else size + design_dim + num_objectives
        ),
        observation_size,
        is_leaf=lambda size: isinstance(size, tuple),
    )


def make_mo_design_mlp_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    design_dim: int,
    num_objectives: int,
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
)->DesignNetworks:
    """Build a conditioned policy MLP and a critic with one output per objective."""
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
        num_objectives=num_objectives,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )

    return DesignNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

def make_mo_design_mlp_inference_fn(networks_: DesignNetworks):
    """Inference-fn factory keyed on the robot design.

    Returns ``inference_fn(params, design, tradeoff, deterministic=False) -> policy(obs, key)``,
    where ``params = (normalizer_params, policy_params)``. ``design`` may be a single
    design ``(design_dim,)`` or batched ``(num_envs, design_dim)``; in the batched case
    obs/params are vmapped over the leading env axis.
    """

    def mo_design_mlp_inference_fn(
        params: types.Params, design: jax.Array, tradeoff: jax.Array, deterministic: bool = False
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
            logits = policy_apply(
                normalizer_params,
                policy_params,
                jax.tree_util.tree_map(
                    lambda obs: jnp.concatenate(
                        (
                            obs,
                            jnp.broadcast_to(design, obs.shape[:-1] + design.shape[-1:]),
                            jnp.broadcast_to(tradeoff, obs.shape[:-1] + tradeoff.shape[-1:]),
                        ),
                        axis=-1,
                    ),
                    observations,
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

    return mo_design_mlp_inference_fn


# ------------------------------------------------------------------------ loss
@flax.struct.dataclass
class DesignMLPParams:
    """Trainable parameters: the design-conditioned policy and value MLPs."""

    policy_params: Params
    value_params: Params

def compute_mo_design_mlp_loss(
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
    num_advantage_groups: int = 1,
    value_loss_fn: Callable = mse_loss,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes the clipped-PPO loss for the multi-objective design MLP.

    Args:
        params: trainable policy and value MLP parameters.
        normalizer_params: observation normalizer params.
        data: ``DesignTransition`` with leading dimensions ``[B, T]``. Its ``design``
            field contains the normalized per-environment design repeated over time and
            requires
            ``extras['state_extras']['truncation']``,
            ``extras['policy_extras']['raw_action']``, and
            ``extras['policy_extras']['log_prob']``.
        rng: PRNG key (for entropy estimate).
        design_networks: the policy/value MLP bundle.
        num_advantage_groups: equal contiguous batch groups for per-objective
            advantage normalization; 1 pools the minibatch.
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
        lambda obs: jnp.concatenate((obs, data.design, data.tradeoff), axis=-1),
        data.observation,
    )
    policy_logits = policy_apply(normalizer_params, policy_params, observation)
    policy_logits = jnp.swapaxes(policy_logits, 0, 1)

    baseline = value_apply(normalizer_params, value_params, observation)
    baseline = jnp.swapaxes(baseline, 0, 1)

    terminal_obs = jax.tree_util.tree_map(
        lambda obs: jnp.concatenate(
            (obs[-1], data.design[-1], data.tradeoff[-1]), axis=-1
        ),
        data.next_observation,
    )
    bootstrap_value = single_value_apply(
        normalizer_params, value_params, terminal_obs
    )

    rewards = data.reward * reward_scaling  # [T, B, num_objectives]

    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = jax.vmap(
        lambda reward, value, bootstrap: ppo_losses.compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=reward,
            values=value,
            bootstrap_value=bootstrap,
            lambda_=gae_lambda,
            discount=discounting,
        ),
        in_axes=(2, 2, 1),
        out_axes=2,
    )(rewards, baseline, bootstrap_value)
    if normalize_advantage:
        # One mean/std per (group, objective): objectives carry different units, so each
        # gets its own scale before the tradeoff mixes them, and B is design-major so the
        # design axis folds out of it into contiguous groups. Standardizing per design
        # keeps a low-return design's share of the policy gradient from shrinking in
        # proportion to how much smaller its returns happen to be.
        grouped = advantages.reshape(
            advantages.shape[0], num_advantage_groups, -1, advantages.shape[2]
        )
        mean = grouped.mean(axis=(0, 2), keepdims=True)
        std = grouped.std(axis=(0, 2), keepdims=True)
        advantages = ((grouped - mean) / (std + 1e-8)).reshape(advantages.shape)

    scalar_advantages = jnp.sum(data.tradeoff * advantages, axis=2)

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * scalar_advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon)
        * scalar_advantages
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
BATCHING_STRATEGIES = ("design", "shuffle", "stratified")


def train_mo_design_mlp(
    environment,
    num_timesteps: int,
    episode_length: int,
    num_parallel_envs: int = 128,
    num_designs: int = 8,
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
    batching_strategy: str = "design",
    strategy: str = "random",
    warmup_strategy: str | None = None,
    warmup_epochs: int = 0,
    rate: float = 0.0,
    # tradeoff sampling
    alpha: float = 1.0,
    sampling: str = "dense",
    network_factory: Callable = make_mo_design_mlp_networks,
    num_evals: int = 10,
    num_eval_envs: int = 64,
    deterministic_eval: bool = True,
    seed: int = 0,
    progress_fn: Callable = lambda *a, **kw: None,
    policy_params_fn: Callable = lambda *a, **kw: None,
    run_evals: bool = True,
    resume: dict | None = None,
    # Accepted for compatibility with minimal-mjx's train (which calls train_fn with
    # these); unused here because this env is model-as-input with its own acting/eval.
    wrap_env_fn: Callable | None = None,
    eval_env=None,
):
    num_cells = num_designs * num_tradeoffs
    assert num_parallel_envs % num_cells == 0, (
        "num_parallel_envs must be divisible by num_designs * num_tradeoffs"
    )
    # Rollouts per cell, which the env budget fixes: the grid fills num_parallel_envs.
    per_cell = num_parallel_envs // num_cells

    assert num_eval_envs % (num_eval_designs * num_eval_tradeoffs) == 0, (
        "num_eval_envs must be divisible by num_eval_designs * num_eval_tradeoffs"
    )

    if batching_strategy not in BATCHING_STRATEGIES:
        raise ValueError(
            f"Unsupported batching_strategy: {batching_strategy!r}; expected one of "
            f"{BATCHING_STRATEGIES}"
        )
    # check constraints
    warmup_strategy = strategy if warmup_strategy is None else warmup_strategy
    for name, value in (("strategy", strategy), ("warmup_strategy", warmup_strategy)):
        if value not in DESIGN_STRATEGIES:
            raise ValueError(
                f"Unsupported {name}: {value!r}; expected one of {DESIGN_STRATEGIES}"
            )
    if "noisy-fixed" in (strategy, warmup_strategy) and rate <= 0.0:
        raise ValueError(
            f"'noisy-fixed' needs rate > 0, got {rate}; at rate 0 the std never leaves "
            "zero and the strategy is just 'fixed'"
        )
    if warmup_strategy != strategy and warmup_epochs <= 0:
        raise ValueError(
            f"warmup_strategy {warmup_strategy!r} differs from strategy {strategy!r} but "
            f"warmup_epochs is {warmup_epochs}, so the warm-up never runs"
        )

    eval_envs_per_cell = num_eval_envs // (num_eval_designs * num_eval_tradeoffs)
    schedule = shared.Schedule.make(
        num_timesteps, num_evals, num_parallel_envs, batch_size, num_minibatches,
        unroll_length, resamples_per_epoch,
    )
    if warmup_strategy != strategy and warmup_epochs >= schedule.num_epochs:
        raise ValueError(
            f"warmup_epochs {warmup_epochs} covers all {schedule.num_epochs} epochs "
            f"(num_evals - 1), so strategy {strategy!r} never runs"
        )
    # A stratified minibatch holds batch_size / num_cells rows of every cell.
    if batching_strategy == "stratified" and batch_size % num_cells:
        raise ValueError(
            f"stratified batching gives every minibatch an equal share of all "
            f"{num_cells} design x tradeoff cells, so batch_size {batch_size} must be a "
            f"multiple of num_designs {num_designs} x num_tradeoffs {num_tradeoffs}"
        )

    key = jax.random.PRNGKey(seed)
    key, key_net = jax.random.split(key)
    grid_rng = np.random.default_rng(seed)

    # Networks and running statistics consume [observation, design, tradeoff].
    num_objectives = len(environment.params.reward.optimization.objectives)
    obs_size = augment_observation_size(
        environment.observation_size, design_dim, num_objectives
    )

    def append_conditioning(data: Grid):
        return jax.tree_util.tree_map(
            lambda obs: jnp.concatenate(
                (obs, data.transitions.design, data.transitions.tradeoff), axis=-1
            ),
            data.transitions.observation,
        )

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
    inference_fn = make_mo_design_mlp_inference_fn(design_networks)
    # The policy appends the design and tradeoff to its observations.
    make_policy = lambda norm, params, designs, tradeoffs, **kw: inference_fn(
        (norm, params.policy_params), designs, tradeoffs, **kw
    )

    optimizer = shared.make_optimizer(learning_rate, max_grad_norm)
    loss_fn = functools.partial(
        compute_mo_design_mlp_loss,
        design_networks       = design_networks,
        entropy_cost          = entropy_cost,
        discounting           = discounting,
        reward_scaling        = reward_scaling,
        gae_lambda            = gae_lambda,
        clipping_epsilon      = clipping_epsilon,
        normalize_advantage   = normalize_advantage,
        # Only a stratified minibatch is guaranteed an equal share of every design;
        # 'design' and 'shuffle' carry no such grouping, so both pool.
        num_advantage_groups  = num_designs if batching_strategy == "stratified" else 1,
        value_loss_fn         = partial(huber_loss, huber_delta = huber_delta) if value_loss_type == 'huber' else mse_loss,
    )
    chunk = shared.make_training_chunk(
        environment, make_policy,
        shared.make_sgd_step(loss_fn, optimizer, num_minibatches, batching_strategy),
        schedule, unroll_length, episode_length, num_updates_per_batch,
        observation_fn=append_conditioning,
    )
    rollout_returns = shared.make_rollout_returns(
        environment, make_policy, episode_length, deterministic_eval
    )
    env_inputs = shared.make_env_inputs(environment)

    anchors = Grid.from_design_sample(environment, seed, num_designs).designs[:, 0]
    design_low, design_high = np.asarray(environment.design_limits, np.float32)
    design_span = design_high - design_low

    def draw(name: str, iteration: int) -> Grid:
        """The grid ``name`` produces ``iteration`` resamples into its own phase.

        ``name`` governs the design axis only; every strategy redraws the ``K`` tradeoffs,
        which ``tradeoff_sampling`` owns.
        """
        if name == "random":
            return Grid.from_uniform_sample(
                environment, grid_rng, num_tradeoffs, num_designs, per_cell,
                sampling=sampling, alpha=alpha,
            )
        designs = anchors
        if name == "noisy-fixed":
            noise = grid_rng.normal(0.0, 1.0, anchors.shape) * (
                rate * iteration * design_span
            )
            designs = np.clip(anchors + noise, design_low, design_high)
        tradeoffs = sample_tradeoffs_cpu(
            grid_rng, num_tradeoffs, num_objectives, sampling=sampling, alpha=alpha
        )
        return Grid.crossed(
            designs, tradeoffs, per_cell, objectives=environment.objectives
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
    eval_grid = Grid.from_uniform_sample(
        environment, seed + 1000, num_eval_tradeoffs, num_eval_designs,
        eval_envs_per_cell, sampling=sampling, alpha=alpha,
    )
    eval_model, eval_designs, eval_tradeoffs = env_inputs(eval_grid)

    def evaluate(training_state, extra_state, key):
        rewards = rollout_returns(
            training_state.normalizer_params,
            training_state.params,
            eval_designs,
            eval_tradeoffs,
            eval_model,
            shared.paired_eval_keys(
                key, num_eval_designs * num_eval_tradeoffs, eval_envs_per_cell
            ),
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
    resume_epoch = 0
    if resume is not None:
        def restore(ts, extra, ckpt):
            # params_of checkpoints the policy alone, so a run whose learner state
            # predates that file cannot get its value network back.
            raise ValueError(
                f"{resume['run_dir']} saved no learner state, and a design_mlp "
                "checkpoint holds only the policy, not the value network PPO trains "
                "against, so this run cannot be continued"
            )

        training_state, _ = shared.resume_training_state(
            resume, training_state, optimizer, restore
        )
        resume_epoch = resume["epoch"]

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
        resume_epoch=resume_epoch,
    )
    return inference_fn, params, metrics


# ----------------------------------------------------------------------- setup
def setup_mo_design_mlp(config):
    """Return ``(train_fn, network_factory)`` for the design-conditioned MLP.
    """
    lp                = config["learning_params"]
    ppo               = dict(lp["ppo_params"])
    net               = dict(lp["network_params"])
    codesign          = dict(config['env_config']['codesign'])
    design_sampling   = dict(lp.get('design_sampling', {}))
    tradeoff_sampling = dict(lp['tradeoff_sampling'])

    for group, name in ((design_sampling, "design_sampling"),
                        (tradeoff_sampling, "tradeoff_sampling")):
        if "per_cell" in group:
            raise ValueError(
                f"{name} 'per_cell' is now derived as num_parallel_envs // (num_designs "
                "* num_tradeoffs), which is what the schedule arithmetic requires; drop "
                "the key and set num_parallel_envs to the grid you want"
            )
    if "sampling" in design_sampling:
        raise ValueError(
            "design_sampling 'sampling' is now 'strategy', optionally preceded by a "
            "'warmup_strategy' for the first 'warmup_epochs' epochs; the simplex "
            "sampler stays under tradeoff_sampling 'sampling'"
        )

    network_factory = functools.partial(
        make_mo_design_mlp_networks,
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
    )

    train_fn = functools.partial(
        train_mo_design_mlp,
        network_factory       = network_factory,
        design_dim            = len(codesign["low"]),
        num_designs           = design_sampling.get("num_designs", 8),
        resamples_per_epoch   = design_sampling.get("resamples_per_epoch", 1),
        strategy              = design_sampling.get("strategy", "random"),
        warmup_strategy       = design_sampling.get("warmup_strategy"),
        warmup_epochs         = design_sampling.get("warmup_epochs", 0),
        rate                  = design_sampling.get("rate", 0.0),
        num_tradeoffs         = tradeoff_sampling["num_tradeoffs"],
        alpha                 = tradeoff_sampling["alpha"],
        sampling              = tradeoff_sampling["sampling"],
        resume                = shared.resume_config(lp),
        **ppo,
    )
    return train_fn, network_factory
