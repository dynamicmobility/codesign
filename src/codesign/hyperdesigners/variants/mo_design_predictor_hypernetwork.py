"""``mo_design_predictor_hypernetwork``: networks, losses and training algo.

:mod:`mo_design_hypernetwork` with a design predictor ``f(d | w)`` that proposes the
designs each tradeoff is trained on, itself trained by GRPO on the returns those designs
earn under the current ``H(d, w)``.
"""

import functools
from functools import partial
from typing import Callable, Literal, NamedTuple, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import distribution, gradients, networks, types
from brax.training.acme import running_statistics
from brax.training.acme.types import NestedArray
from brax.training.types import Params, PRNGKey
from flax import linen

from codesign.hyperdesigners import shared
from codesign.hyperdesigners.losses import (
    DesignHypernetParams,
    huber_loss,
    mse_loss,
)
from codesign.hyperdesigners.variants.mo_design_hypernetwork import (
    compute_mo_design_hypernet_loss,
    make_mo_design_hypernet_networks,
    make_mo_design_inference_fn,
)
from codesign.hyperdesigners.networks import FeedForwardHypernetwork
from codesign.utils import model as model_lib
from codesign.utils.grid import Grid, sample_tradeoffs_cpu


# --------------------------------------------------------------------------- networks
@flax.struct.dataclass
class DesignPredictorHypernetNetworks:
    hypernetwork: FeedForwardHypernetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    design_predictor_network: networks.FeedForwardNetwork
    parametric_design_distribution: distribution.ParametricDistribution


def make_mo_design_predictor_hypernet_networks(
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
    # Design Hypernetwork
    design_distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    design_hidden_layer_sizes: Sequence[int] = (16,) * 2,
    design_noise_std_type: Literal["scalar", "log"] = "scalar",
    design_init_noise_std: float = 1.0,

) -> DesignPredictorHypernetNetworks:
    """Build a hypernetwork conditioned on ``[design, tradeoff]``: ``H(d, w)`` alongside a design predictor.
    The hypernetwork/value/action are the same as make_design_hypernet_networks, but the design predictor
    network is also included.

    The design predictor network is a feedforward network which outputs a parametric distribution

    """
    design_hypernet = make_mo_design_hypernet_networks(
        observation_size=observation_size,
        action_size=action_size,
        design_dim=design_dim,
        num_objectives=num_objectives,
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
    )

    if design_distribution_type == "normal":
        parametric_design_distribution = distribution.NormalDistribution(
            event_size=design_dim
        )
    elif design_distribution_type == "tanh_normal":
        parametric_design_distribution = distribution.NormalTanhDistribution(
            event_size=design_dim
        )
    else:
        raise ValueError(
            f'Unsupported distribution type: {design_distribution_type}. Must be one'
            ' of "normal" or "tanh_normal".'
        )

    # The "observation" is the tradeoff, which already lies on the simplex, so
    # preprocess_observations_fn is left at brax's identity default. noise_std_type /
    # init_noise_std / state_dependent_std are only read by the "normal" branch of
    # make_policy_network; under "tanh_normal" the std comes from the MLP head.
    design_predictor_network = networks.make_policy_network(
        param_size=parametric_design_distribution.param_size,
        obs_size=num_objectives,
        hidden_layer_sizes=design_hidden_layer_sizes,
        activation=activation,
        distribution_type=design_distribution_type,
        noise_std_type=design_noise_std_type,
        init_noise_std=design_init_noise_std,
        state_dependent_std=state_dependent_std,
    )
    return DesignPredictorHypernetNetworks(
        hypernetwork = design_hypernet.hypernetwork,
        policy_network = design_hypernet.policy_network,
        value_network = design_hypernet.value_network,
        parametric_action_distribution = design_hypernet.parametric_action_distribution,
        design_predictor_network = design_predictor_network,
        parametric_design_distribution = parametric_design_distribution
    )


def make_design_predictor_inference_fn(networks_: DesignPredictorHypernetNetworks):
    """Inference-fn factory for the design predictor ``f(d | w)``.

    Returns ``design_predictor_inference_fn(params, tradeoffs, deterministic=False,
    key_sample=None) -> (designs, extras)``, where ``params`` is the design-predictor
    network's params alone (its input is a simplex tradeoff, so there is no normalizer)
    and ``designs`` are normalized to ``[0, 1]``. ``tradeoffs`` may be a single vector
    ``(num_objectives,)`` or batched ``(n, num_objectives)``.

    ``extras['raw_action']`` is the *pre-tanh* sample, which is what
    ``parametric_design_distribution.log_prob`` expects; the returned design is the tanh
    output mapped from ``(-1, 1)`` onto ``[0, 1]``.
    """

    def design_predictor_inference_fn(
            params: types.Params,
            tradeoffs: jax.Array,
            deterministic: bool = True,
            key_sample: PRNGKey = None,
    ):
        design_predictor_network = networks_.design_predictor_network
        parametric_design_distribution = networks_.parametric_design_distribution
        # No preprocessing, so the processor params are unused.
        logits = design_predictor_network.apply(None, params, tradeoffs)

        if deterministic:
            return 0.5 * (parametric_design_distribution.mode(logits) + 1.0), {}
        raw_actions = parametric_design_distribution.sample_no_postprocessing(
            logits, key_sample
        )
        # raw_actions ranges from -1 to 1 and a transformation is applied to keep it in the range
        # of 0 to 1. 
        # TODO: We can remove the need to do this by keeping the "unnormalized"
        # range from -1 to 1, but will need to change this in multiple places
        log_prob = parametric_design_distribution.log_prob(logits, raw_actions)
        designs = 0.5 * (parametric_design_distribution.postprocess(raw_actions) + 1.0)
        return designs, {
            "log_prob": log_prob,
            "raw_action": raw_actions,
        }

    return design_predictor_inference_fn


# ---------------------------------------------------------------------- losses
class DesignPredictorTransition(NamedTuple):
    """One GRPO group per row: tradeoff ``w_t`` with ``G`` designs drawn from ``f(. | w_t)``."""

    tradeoff        : NestedArray # (n_tradeoffs, num_objectives)
    raw_design      : NestedArray # pre-tanh samples of f (n_tradeoffs, G, design_dim)
    value           : NestedArray # discounted scalarized returns (n_tradeoffs, G)
    design_log_prob : NestedArray # log f(raw_design | w) at sample time (n_tradeoffs, G)


def compute_grpo_loss(
    design_predictor_params: types.Params,
    data: DesignPredictorTransition,
    rng: jnp.ndarray,
    design_networks: DesignPredictorHypernetNetworks,
    entropy_cost: float = 1e-4,
    clipping_epsilon: float = 0.3,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes the clipped GRPO loss for the design predictor ``f(d | w)``.

    Args:
        design_predictor_params: trainable design-predictor params. Must stay first so
            ``gradients.gradient_update_fn`` differentiates w.r.t. it.
        data: ``DesignPredictorTransition`` with a leading group axis.
        rng: PRNG key (for the entropy estimate).
        design_networks: the design predictor hypernetwork bundle.
    """
    parametric_design_distribution = design_networks.parametric_design_distribution

    # One logit row per tradeoff; no preprocessing, so the processor params are unused.
    logits = design_networks.design_predictor_network.apply(
        None, design_predictor_params, data.tradeoff
    )

    # Every design in a group was drawn from the same tradeoff, so it shares its logits.
    group_logits = jnp.repeat(
        logits[:, None], data.raw_design.shape[1], axis=1
    )
    # log_prob expects the pre-tanh sample: it subtracts the tanh jacobian itself.
    target_design_log_prob = parametric_design_distribution.log_prob(
        group_logits, data.raw_design
    )
    # Policy ratio rho between the updated and the sampling-time distribution.
    rho_s = jnp.exp(target_design_log_prob - data.design_log_prob)

    # Group-relative advantage: standardize within each tradeoff's own group. A group
    # whose designs all scored the same carries no ranking information; its std is
    # float-rounding noise, which the division would blow up into a +-1 advantage.
    r_mean = jnp.mean(data.value, axis=1, keepdims=True)
    r_std = jnp.std(data.value, axis=1, keepdims=True)
    degenerate = r_std <= 1e-6 * jnp.abs(r_mean)  # 1e-6 ~ 10x float32 eps
    advantages = jnp.where(
        degenerate, 0.0, (data.value - r_mean) / (r_std + 1e-8)
    )

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Entropy bonus, on the unrepeated logits (the repeats are identical).
    entropy = jnp.mean(parametric_design_distribution.entropy(logits, rng))
    entropy_loss = entropy_cost * -entropy

    total_loss = policy_loss + entropy_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "entropy_loss": entropy_loss,
        "design_entropy": entropy,
    }


# -------------------------------------------------------------------- training
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
    network_factory: Callable = make_mo_design_predictor_hypernet_networks,
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
    assert 0 <= num_warmup_iters <= schedule.num_epochs, (
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
    inference_fn = make_mo_design_inference_fn(design_networks)
    design_predictor_inference_fn = make_design_predictor_inference_fn(
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
        value_loss_fn         = partial(huber_loss, huber_delta = huber_delta) if value_loss_type == 'huber' else mse_loss,
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
        key,
        inference_fn,
        env_inputs,
        extra_state=design_predictor_state,
        run_evals=run_evals,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    return inference_fn, params, metrics


# ----------------------------------------------------------------------- setup
def setup_mo_design_predictor_hypernetwork(config):
    """Return ``(train_fn, network_factory)`` for the MO design predictor hypernetwork.

    ``num_designs`` is the GRPO group size: the number of designs drawn from ``f(d | w)``
    for each tradeoff.
    """
    lp                = config["learning_params"]
    ppo               = dict(lp["ppo_params"])
    net               = dict(lp["network_params"])
    codesign          = dict(config['env_config']['codesign'])
    design_sampling   = dict(lp['design_sampling'])
    tradeoff_sampling = dict(lp['tradeoff_sampling'])
    predictor         = dict(lp['design_predictor'])

    # Network kwargs must carry defaults on the factory so brax's checkpoint config
    # records them and get_network can rebuild the bundle at load time.
    network_factory = functools.partial(
        make_mo_design_predictor_hypernet_networks,
        hypersize                   = tuple(net["hypersize"]),
        num_features                = net["num_features"],
        policy_hidden_layer_sizes   = tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes    = tuple(net["value_hidden_layer_sizes"]),
        design_hidden_layer_sizes   = tuple(predictor["hidden_layer_sizes"]),
    )

    OPTIONAL_ARGS = ( # TODO: move these into the required args when mature
        (predictor,         "design_learning_rate"),
        (predictor,         "design_entropy_cost"),
        (predictor,         "design_clipping_epsilon"),
        (predictor,         "num_design_updates_per_batch"),
        (predictor,         "num_warmup_iters"),
        (design_sampling,   "num_eval_designs"),
        (tradeoff_sampling, "num_eval_tradeoffs"),
    )
    optional_params = {a: group[a] for group, a in OPTIONAL_ARGS if a in group}

    train_fn = functools.partial(
        train_mo_design_predictor,
        network_factory              = network_factory,
        design_dim                   = len(codesign["low"]),
        num_designs                  = design_sampling["num_designs"],
        resamples_per_epoch          = design_sampling["resamples_per_epoch"],
        num_tradeoffs                = tradeoff_sampling["num_tradeoffs"],
        alpha                        = tradeoff_sampling["alpha"],
        sampling                     = tradeoff_sampling["sampling"],
        **optional_params,
        **ppo,
    )
    return train_fn, network_factory
