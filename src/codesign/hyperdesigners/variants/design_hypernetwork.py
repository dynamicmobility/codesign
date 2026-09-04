"""``design_hypernetwork``: networks, loss and training algo.

A hypernetwork maps a robot design to the weights of a policy MLP and a value MLP,
trained on brax's clipped-PPO loss. The multi-objective algos build on the bundle and
builders defined here.
"""

import functools
from functools import partial
from typing import Any, Callable, Literal, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from brax.training import networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, losses as ppo_losses
from brax.training.networks import Initializer
from brax.training.types import PRNGKey
from etils import epath
from flax import linen

from codesign.hyperdesigners.hypernetworks import (
    MLP,
    DualA2CHypernet,
    sobol_design_table,
)

from codesign.hyperdesigners import shared
from codesign.hyperdesigners.acting import DesignTransition
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    huber_loss,
    mse_loss,
)
from codesign.hyperdesigners.networks import (
    DesignHypernetNetworks,
    FeedForwardHypernetwork,
    make_target_networks,
    resolve_kernel_initializer,
)
from codesign.utils.grid import Grid


# --------------------------------------------------------------------------- networks
HypernetInitStrategy = Literal["bias", "weight", "load_network"]


def make_design_hypernetwork(
    design_dim: int,
    obs_dim: int,
    target_policy_dict: dict,
    target_value_dict: dict,
    hypersize: tuple,
    num_features: int = 8,
    w_variance: float = 0.0,
    initialization_strategy: HypernetInitStrategy = "bias",
    weight_initializer: str | Initializer = "kaiming_uniform",
) -> FeedForwardHypernetwork:
    """Wrap a ``DualA2CHypernet`` keyed on the design (no simplex normalization).

    ``apply(params, design) -> (policy_params, value_params)`` where each is a Flax
    ``{'params': ...}`` tree (batched along axis 0 when ``design`` is batched).
    """
    weight_initializer = resolve_kernel_initializer(weight_initializer)
    if(initialization_strategy not in ("bias", "weight", "load_network")):
        raise ValueError(f"Unsupported initialization_strategy: {initialization_strategy!r}")

    hypernet = DualA2CHypernet(
        target_policy_dict=target_policy_dict,
        target_value_dict=target_value_dict,
        num_objs=design_dim,
        obs_dim=obs_dim,
        hypersize=hypersize,
        num_features=num_features,
        W_variance=w_variance,
    )

    if(initialization_strategy == "weight"):
        def init(key):
            key_hypernet, key_policy_w, key_value_w = jax.random.split(key, 3)
            params = flax.core.unfreeze(hypernet.init(key_hypernet, dummy_design))
            params["params"]["policy_b"] = jnp.zeros_like(
                params["params"]["policy_b"]
            )
            params["params"]["value_b"] = jnp.zeros_like(
                params["params"]["value_b"]
            )
            policy_w = params["params"]["policy_W"]
            value_w = params["params"]["value_W"]
            params["params"]["policy_W"] = weight_initializer(
                key_policy_w, policy_w.shape, policy_w.dtype
            )
            params["params"]["value_W"] = weight_initializer(
                key_value_w, value_w.shape, value_w.dtype
            )
            return flax.core.freeze(params)
    else:
        def init(key):
            return hypernet.init(key, dummy_design)

    dummy_design = jnp.zeros(design_dim)

    def apply(params, design):
        # Returns ((policy_params, value_params), (flat...), (features...)); take [0].
        return hypernet.apply(params, design)[0]

    return FeedForwardHypernetwork(init=init, apply=apply)


def make_design_hypernet_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    design_dim: int,
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
    num_value_outputs: int = 1,
    initialization_strategy: HypernetInitStrategy = 'bias',
    weight_initializer: str | Initializer = "kaiming_uniform",
) -> DesignHypernetNetworks:
    """Build the target policy/value MLPs and the design-conditioned hypernetwork."""
    (
        parametric_action_distribution,
        policy_network,
        value_network,
        target_policy_params,
        target_value_params,
        obs_dim,
    ) = make_target_networks( # TODO: make this return a named tuple?
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

    hypernetwork = make_design_hypernetwork(
        design_dim=design_dim,
        obs_dim=obs_dim,
        target_policy_dict=target_policy_params,
        target_value_dict=target_value_params,
        hypersize=hypersize,
        num_features=num_features,
        w_variance=w_variance,
        initialization_strategy=initialization_strategy,
        weight_initializer=weight_initializer,
    )

    return DesignHypernetNetworks(
        hypernetwork=hypernetwork,
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_design_inference_fn(networks_: DesignHypernetNetworks):
    """Inference-fn factory keyed on the robot design.

    Returns ``inference_fn(params, design, deterministic=False) -> policy(obs, key)``,
    where ``params = (normalizer_params, hypernet_params)``. ``design`` may be a single
    design ``(design_dim,)`` or batched ``(num_envs, design_dim)``; in the batched case
    obs/params are vmapped over the leading env axis.
    """

    def design_inference_fn(
        params: types.Params, design: jax.Array, deterministic: bool = False
    ) -> types.Policy:
        normalizer_params, hypernet_params = params[0], params[1]
        policy_network = networks_.policy_network
        parametric_action_distribution = networks_.parametric_action_distribution

        # Policy params from the hypernetwork (value head is ignored at acting time).
        policy_params, _ = networks_.hypernetwork.apply(hypernet_params, design)

        if len(design.shape) == 1:
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

    return design_inference_fn

# ------------------------------------------------------------------------ loss
def compute_design_hypernet_loss(
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
    value_loss_fn: Callable = mse_loss,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes the clipped-PPO loss for the design hypernetwork.

    Args:
        params: trainable hypernetwork params.
        normalizer_params: observation normalizer params.
        data: ``DesignTransition`` with leading dims ``[B, T]``. Requires
            ``extras['state_extras']['truncation']``,
            ``extras['policy_extras']['raw_action']``,
            ``extras['policy_extras']['log_prob']``.
        rng: PRNG key (for entropy estimate).
        design_networks: the design hypernetwork bundle.
    """
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

    # Per-env policy/value params from the hypernetwork (design is constant over time).
    policy_params, value_params = design_networks.hypernetwork.apply(
        params.hypernetwork, data.design[:, 0]
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



# ------------------------------------------------------------------------ handoff
def _fit_features(mlp, params, designs, targets, steps: int, learning_rate: float):
    """Least-squares fit of a feature MLP to ``targets`` at ``designs``.

    Returns ``(params, final_mse)``.
    """
    optimizer = optax.adam(learning_rate)
    loss = lambda p: jnp.mean((mlp.apply({"params": p}, designs) - targets) ** 2)

    def step(carry, _):
        p, state = carry
        value, grad = jax.value_and_grad(loss)(p)
        updates, state = optimizer.update(grad, state)
        return (optax.apply_updates(p, updates), state), value

    (params, _), losses = jax.jit(
        lambda p: jax.lax.scan(step, (p, optimizer.init(p)), (), length=steps)
    )(params)
    return params, float(losses[-1])


def load_warmup_params(
    training_state,
    checkpoint_path: str,
    optimizer,
    design_limits,
    fit_steps: int = 2000,
    fit_learning_rate: float = 1e-3,
    quiet: bool = False,
):
    """Seed this run from a ``design_lookup_hypernetwork`` warm-up checkpoint.

    Uses the warm-up W and distills the one hot encoding into the task encoder
    MLP so that behavior is replicated at the original sampled designs from
    the lookup table in design_lookup_hypernetwork.

    Args:
        training_state: this run's freshly initialized state.
        checkpoint_path: a warm-up checkpoint step directory.
        optimizer: the optimizer whose state is re-initialized on the merged params.
        design_limits: this run's design box, ``(2, design_dim)`` lows then highs.
        fit_steps, fit_learning_rate: Adam budget for the one-hot regression.

    Returns:
        The updated ``TrainingState``.
    """
    path = epath.Path(checkpoint_path)
    warmup = yaml.safe_load((path.parent / "config.yaml").read_text())
    box = warmup["env_config"]["codesign"]
    # The anchors are normalized by the warm-up's box and this run feeds normalized
    # designs, so a different box points the same coordinate at a different robot.
    limits, warm_limits = (
        np.asarray(design_limits, np.float32),
        np.asarray([box["low"], box["high"]], np.float32),
    )
    if limits.shape != warm_limits.shape or not np.allclose(limits, warm_limits):
        raise ValueError(
            f"the warm-up's design box is {warm_limits.tolist()} but this run's is "
            f"{limits.tolist()}; the same normalized design would be a different robot"
        )
    table = jnp.asarray(sobol_design_table(
        warmup["learning_params"]["ppo_params"]["seed"],
        warmup["learning_params"]["design_sampling"]["num_designs"],
        box["low"], box["high"],
    ))

    params = flax.core.unfreeze(training_state.params.hypernetwork)["params"]
    # Layer widths come off the fresh tree, so the warm-up needs no extra plumbing.
    mlp_params = params["policy_mlp"]
    widths = tuple(
        mlp_params[f"Dense_{i}"]["bias"].shape[0] for i in range(len(mlp_params))
    )
    hypersize, num_features = widths[:-1], widths[-1]
    if len(table) != num_features:
        raise ValueError(
            f"the warm-up has {len(table)} designs but this run has num_features="
            f"{num_features}; a transfer needs num_features == the warm-up's num_designs"
        )

    normalizer_params, warm = checkpoint.load(str(path.resolve()))
    warm = flax.core.unfreeze(warm)["params"]
    warm_net = warmup["learning_params"]["network_params"]
    for name in ("policy_W", "policy_b", "value_W", "value_b"):
        # A row of W is a flattened policy, so its width encodes the hidden sizes.
        if params[name].shape != warm[name].shape:
            raise ValueError(
                f"{name} is {warm[name].shape} in the warm-up but {params[name].shape} "
                f"here; this run must use the warm-up's hidden sizes, policy "
                f"{warm_net['policy_hidden_layer_sizes']} and value "
                f"{warm_net['value_hidden_layer_sizes']}"
            )
        params[name] = warm[name]

    targets = jnp.eye(num_features)
    mlp = MLP(obs_dim=table.shape[-1], hidden_sizes=hypersize, action_dim=num_features)
    for name in ("policy_mlp", "value_mlp"):
        params[name], mse = _fit_features(
            mlp, params[name], table, targets, fit_steps, fit_learning_rate
        )
        if not quiet:
            print(f"load_network: {name} one-hot fit MSE {mse:.3e}")

    merged = DesignHypernetParams(hypernetwork=flax.core.freeze({"params": params}))
    return training_state.replace(
        params=merged,
        normalizer_params=normalizer_params,
        optimizer_state=optimizer.init(merged),
    )



# -------------------------------------------------------------------- training
def train_design_hypernetwork(
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
    per_cell: int = 16, # How many times each design should be trialed
    resamples_per_epoch: int = 1,
    network_factory: Callable = make_design_hypernet_networks,
    num_evals: int = 10,
    num_eval_envs: int = 64,
    deterministic_eval: bool = True,
    seed: int = 0,
    progress_fn: Callable = lambda *a: None,
    policy_params_fn: Callable = lambda *a: None,
    run_evals: bool = True,
    batching_strategy: str = 'shuffle',
    warmup_checkpoint: str | None = None,
    # Accepted for compatibility with minimal-mjx's train (which calls train_fn with
    # these); unused here because this env is model-as-input with its own acting/eval.
    wrap_env_fn: Callable | None = None,
    eval_env=None,
):
    assert (num_designs * per_cell) % num_parallel_envs == 0, (
        "total number of environments (num_designs*per_cell) must be divisible by num_parallel_envs"
    )
    assert num_eval_envs % num_designs == 0, (
        "num_eval_envs must be divisible by num_designs"
    )
    # Batching is by design, so a minibatch is one design's rollouts only at equality.
    assert num_minibatches == num_designs, (
        "num_minibatches must equal num_designs for one design per minibatch"
    )
    schedule = shared.Schedule.make(
        num_timesteps, num_evals, num_parallel_envs, batch_size, num_minibatches,
        unroll_length, resamples_per_epoch,
    )

    key = jax.random.PRNGKey(seed)
    key, key_net = jax.random.split(key)
    design_rng = np.random.default_rng(seed)

    normalize = (
        running_statistics.normalize if normalize_observations else (lambda x, y: x)
    )
    design_networks = network_factory(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        design_dim=design_dim,
        key=key_net,
        preprocess_observations_fn=normalize,
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
        shared.make_sgd_step(loss_fn, optimizer, num_minibatches, batching_strategy),
        schedule, unroll_length, episode_length, num_updates_per_batch,
    )
    rollout_returns = shared.make_rollout_returns(
        environment, make_policy, episode_length, deterministic_eval
    )
    env_inputs = shared.make_env_inputs(environment)

    def sample(it, extra_state, key):
        """``num_designs`` designs, tiled across the envs, against the trivial tradeoff."""

        return Grid.from_design_sample( environment, design_rng, num_designs, per_cell=per_cell), None

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

    params_of = lambda ts, extra: (ts.normalizer_params, ts.params.hypernetwork)

    training_state = shared.init_training_state(
        DesignHypernetParams(hypernetwork=design_networks.hypernetwork.init(key_net)),
        optimizer, environment.observation_size,
    )
    if warmup_checkpoint is not None:
        training_state = load_warmup_params(
            training_state, warmup_checkpoint, optimizer, environment.design_limits
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
def setup_design_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the design hypernetwork.
    """
    lp = config["learning_params"]
    ppo = dict(lp["ppo_params"])
    net = dict(lp["network_params"])
    design = dict(config['env_config']['codesign'])
    design_sampling = dict(lp.get("design_sampling", {}))

    strategy = net.get("initialization_strategy", "bias")
    warmup_checkpoint = net.get("warmup_checkpoint")
    if strategy == "load_network" and warmup_checkpoint is None:
        raise ValueError(
            "initialization_strategy 'load_network' needs a 'warmup_checkpoint' path in "
            "network_params, pointing at a design_lookup_hypernetwork checkpoint"
        )

    network_factory = functools.partial(
        make_design_hypernet_networks,
        hypersize                   = tuple(net["hypersize"]),
        num_features                = net["num_features"],
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
        initialization_strategy     = strategy,
        weight_initializer          = net.get("weight_initializer", "kaiming_uniform"),
    )

    train_fn = functools.partial(
        train_design_hypernetwork,
        network_factory     = network_factory,
        design_dim          = len(design['low']),
        num_designs         = design_sampling.get("num_designs", 8),
        per_cell            = design_sampling.get("per_cell", 16),
        resamples_per_epoch = design_sampling.get("resamples_per_epoch", 1),
        warmup_checkpoint   = warmup_checkpoint if strategy == "load_network" else None,
        **ppo,
    )
    return train_fn, network_factory

