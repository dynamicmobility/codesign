"""``mo_design_hypernetwork``: networks, loss and training algo.

Widens :mod:`design_hypernetwork`'s conditioning to ``[design, tradeoff]`` and scalarizes
a vector-valued critic against the tradeoff. The design-predictor algo builds on the
builders and loss defined here.
"""

from __future__ import annotations

import functools
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from brax.training import networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.types import PRNGKey
from flax import linen

from codesign.hyperdesigners import shared
from codesign.hyperdesigners.acting import DesignTransition
from codesign.hyperdesigners.variants.design_hypernetwork import (
    BATCHING_STRATEGIES,
    DESIGN_STRATEGIES,
    make_design_hypernet_networks,
)
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    huber_loss,
    mse_loss,
)
from codesign.hyperdesigners.networks import DesignHypernetNetworks
from codesign.utils.grid import Grid, sample_tradeoffs_cpu

if TYPE_CHECKING:
    # The predictor bundle is defined downstream, in mo_design_predictor_hypernetwork.
    from codesign.hyperdesigners.variants.mo_design_predictor_hypernetwork import (
        DesignPredictorHypernetNetworks,
    )


# --------------------------------------------------------------------------- networks
def make_mo_design_hypernet_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    design_dim: int,
    num_objectives: int,
    key: jax.Array,
    hypersize: tuple = (128, 128),
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
    num_features: int = 8,
    w_variance: float = 0.0,
) -> DesignHypernetNetworks:
    """Build a hypernetwork conditioned on ``[design, tradeoff]``: ``H(d, w)``.

    Structurally identical to :func:`make_design_hypernet_networks`, except the
    hypernetwork's conditioning input is widened to ``design_dim + num_objectives`` so it
    consumes the concatenation of the (normalized) design and the tradeoff ``w``. The
    returned bundle is the standard :class:`DesignHypernetNetworks`.
    """
    return make_design_hypernet_networks(
        observation_size=observation_size,
        action_size=action_size,
        design_dim=design_dim + num_objectives,
        key=key,
        hypersize=hypersize,
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
        num_features=num_features,
        w_variance=w_variance,
        num_value_outputs=num_objectives,
    )

def make_mo_design_inference_fn(networks_: DesignHypernetNetworks | DesignPredictorHypernetNetworks):
    """Inference-fn factory keyed on ``(design, tradeoff)``.

    Returns ``inference_fn(params, designs, tradeoffs, deterministic=False) ->
    policy(obs, key)``, where ``params = (normalizer_params, hypernet_params)``. ``designs``
    (normalized to ``[0, 1]``) and ``tradeoffs`` (simplex tradeoffs) may be single vectors
    or batched ``(num_envs, ...)``; they are concatenated along the last axis before the
    hypernetwork is applied.
    """

    def mo_design_inference_fn(
        params: types.Params,
        designs: jax.Array,
        tradeoffs: jax.Array,
        deterministic: bool = False,
    ) -> types.Policy:
        """Returns a multi-objective design-conditioned policy hypernetwork function."""
        normalizer_params, hypernet_params = params[0], params[1]
        policy_network = networks_.policy_network
        parametric_action_distribution = networks_.parametric_action_distribution

        cond = jnp.concatenate([designs, tradeoffs], axis=-1)
        # Policy params from the hypernetwork (value head is ignored at acting time).
        policy_params, _ = networks_.hypernetwork.apply(hypernet_params, cond)

        if len(cond.shape) == 1:
            policy_apply = policy_network.apply
        else:
            policy_apply = jax.vmap(policy_network.apply, in_axes=(None, 0, 0))

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = policy_apply(normalizer_params, policy_params, observations)
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

    return mo_design_inference_fn

# ------------------------------------------------------------------------ loss
def compute_mo_design_hypernet_loss(
    params: DesignHypernetParams,
    normalizer_params: Any,
    data: DesignTransition,
    rng: jnp.ndarray,
    design_networks: DesignHypernetNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    num_advantage_groups: int = 1,
    value_loss_fn: Callable = mse_loss,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes the clipped-PPO loss for the multi-objective design hypernetwork ``H(d, w)``.

    Args:
        params: trainable hypernetwork params.
        normalizer_params: observation normalizer params.
        data: ``DesignTransition`` with leading dims ``[B, T]``. ``reward``/``tradeoff``
            carry a trailing objective axis of size ``M``. Requires
            ``extras['state_extras']['truncation']``,
            ``extras['policy_extras']['raw_action']``,
            ``extras['policy_extras']['log_prob']``.
        rng: PRNG key (for entropy estimate).
        design_networks: the design hypernetwork bundle.
        num_advantage_groups: equal contiguous groups ``B`` splits into for advantage
            standardization; ``1`` pools the minibatch.
    """
    designs, tradeoffs = data.design, data.tradeoff
    parametric_action_distribution = design_networks.parametric_action_distribution
    policy_apply = jax.vmap(
        design_networks.policy_network.apply, in_axes=(None, 0, 1)
    )
    value_apply = jax.vmap(
        design_networks.value_network.apply, in_axes=(None, 0, 1)
    )
    single_value_apply = jax.vmap(
        design_networks.value_network.apply, in_axes=(None, 0, 0)
    )

    # Per-env policy/value params from the hypernetwork. Design and tradeoff are constant
    # over time, so use their value at the first timestep.
    cond = jnp.concatenate([designs, tradeoffs], axis=-1)
    policy_params, value_params = design_networks.hypernetwork.apply(
        params.hypernetwork, cond[:, 0]
    )

    # Put the time dimension first: [B, T, ...] -> [T, B, ...].
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    policy_logits = policy_apply(normalizer_params, policy_params, data.observation)
    policy_logits = jnp.swapaxes(policy_logits, 0, 1)

    baseline = value_apply(normalizer_params, value_params, data.observation)
    baseline = jnp.swapaxes(baseline, 0, 1)

    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = single_value_apply(
        normalizer_params, value_params, terminal_obs
    )

    rewards = data.reward * reward_scaling

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
def train_mo_design_hypernetwork(
    environment,
    num_timesteps: int,
    episode_length: int,
    num_parallel_envs: int = 1024,
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
    batching_strategy: str = 'design',
    strategy: str = 'random',
    warmup_strategy: str | None = None,
    warmup_epochs: int = 0,
    rate: float = 0.0,
    # tradeoff sampling
    alpha: float = 1.0,
    sampling: str = "dense",
    network_factory: Callable = make_mo_design_hypernet_networks,
    num_evals: int = 10,
    num_eval_envs: int = 64,
    deterministic_eval: bool = True,
    seed: int = 0,
    progress_fn: Callable = lambda *a, **kw: None,
    policy_params_fn: Callable = lambda *a, **kw: None,
    run_evals: bool = True,
    resume: dict | None = None,
    # Accepted for compatibility with minimal-mjx's train (unused here).
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
    inference_fn = make_mo_design_inference_fn(design_networks)
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
        # Only a stratified minibatch is guaranteed an equal share of every design;
        # 'design' and 'shuffle' carry no such grouping, so both pool.
        num_advantage_groups  = num_designs if batching_strategy == "stratified" else 1,
        value_loss_fn         = partial(huber_loss, huber_delta = huber_delta) if value_loss_type == 'huber' else mse_loss,
    )
    chunk = shared.make_training_chunk(
        environment, make_policy,
        shared.make_sgd_step(loss_fn, optimizer, num_minibatches, batching_strategy),
        schedule, unroll_length, episode_length, num_updates_per_batch,
    )
    rollout_returns = shared.make_rollout_returns(
        environment, make_policy, episode_length, deterministic_eval
    )
    env_inputs = shared.make_env_inputs(environment)

    # The anchors 'fixed' holds and 'noisy-fixed' jitters, (num_designs, design_dim) in
    # physical units.
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
        """Designs crossed with freshly sampled tradeoffs."""
        nonlocal phase, iteration
        # iterations start counting when sampling strategy begins
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

    params_of = lambda ts, extra: (ts.normalizer_params, ts.params.hypernetwork)

    training_state = shared.init_training_state(
        DesignHypernetParams(hypernetwork=design_networks.hypernetwork.init(key_net)),
        optimizer, environment.observation_size,
    )
    resume_epoch = 0
    if resume is not None:
        training_state, _ = shared.resume_training_state(
            resume, training_state, optimizer,
            lambda ts, extra, ckpt: (
                ts.replace(params=ts.params.replace(hypernetwork=ckpt[1])), extra
            ),
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
def setup_mo_design_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the multi-objective design hypernetwork.
    """
    lp                = config["learning_params"]
    ppo               = dict(lp["ppo_params"])
    net               = dict(lp["network_params"])
    codesign          = dict(config['env_config']['codesign'])
    design_sampling   = dict(lp['design_sampling'])
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
        make_mo_design_hypernet_networks,
        hypersize                   = tuple(net["hypersize"]),
        num_features                = net["num_features"],
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
    )

    OPTIONAL_ARGS = ( # TODO: move these into the required args when mature
        (design_sampling,   "num_eval_designs"),
        (tradeoff_sampling, "num_eval_tradeoffs"),
    )
    optional_params = {a: group[a] for group, a in OPTIONAL_ARGS if a in group}

    train_fn = functools.partial(
        train_mo_design_hypernetwork,
        network_factory       = network_factory,
        design_dim            = len(codesign["low"]),
        num_designs           = design_sampling["num_designs"],
        resamples_per_epoch   = design_sampling["resamples_per_epoch"],
        strategy              = design_sampling.get("strategy", "random"),
        warmup_strategy       = design_sampling.get("warmup_strategy"),
        warmup_epochs         = design_sampling.get("warmup_epochs", 0),
        rate                  = design_sampling.get("rate", 0.0),
        num_tradeoffs         = tradeoff_sampling["num_tradeoffs"],
        alpha                 = tradeoff_sampling["alpha"],
        sampling              = tradeoff_sampling["sampling"],
        resume                = shared.resume_config(lp),
        **optional_params,
        **ppo,
    )
    return train_fn, network_factory

