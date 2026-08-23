import dataclasses
from typing import Any, Callable, Literal, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.networks import ActivationFn, DistributionalCritic, Initializer, FeedForwardNetwork, MLP, Mapping
from brax.training.networks import normalizer_select, _get_obs_state_size
from brax.training.types import PRNGKey
from flax import linen

from moplayground.moppo.networks import DualA2CHypernet


@dataclasses.dataclass
class FeedForwardHypernetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


@flax.struct.dataclass
class DesignHypernetNetworks:
    hypernetwork: FeedForwardHypernetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution

@flax.struct.dataclass
class DesignPredictorHypernetNetworks:
    hypernetwork: FeedForwardHypernetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    design_predictor_network: networks.FeedForwardNetwork
    parametric_design_distribution: distribution.ParametricDistribution


def make_design_hypernetwork(
    design_dim: int,
    obs_dim: int,
    target_policy_dict: dict,
    target_value_dict: dict,
    hypersize: tuple,
    num_features: int = 8,
    w_variance: float = 0.0,
) -> FeedForwardHypernetwork:
    """Wrap a ``DualA2CHypernet`` keyed on the design (no simplex normalization).

    ``apply(params, design) -> (policy_params, value_params)`` where each is a Flax
    ``{'params': ...}`` tree (batched along axis 0 when ``design`` is batched).
    """
    hypernet = DualA2CHypernet(
        target_policy_dict=target_policy_dict,
        target_value_dict=target_value_dict,
        num_objs=design_dim,
        obs_dim=obs_dim,
        hypersize=hypersize,
        num_features=num_features,
        W_variance=w_variance,
    )

    dummy_design = jnp.zeros(design_dim)

    def init(key):
        return hypernet.init(key, dummy_design)

    def apply(params, design):
        # Returns ((policy_params, value_params), (flat...), (features...)); take [0].
        return hypernet.apply(params, design)[0]

    return FeedForwardHypernetwork(init=init, apply=apply)


def make_vector_value_network(
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    num_objectives: int = 1,
    activation: ActivationFn = linen.relu,
    obs_key: str = 'state',
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    use_distributional_critic: bool = False,
    num_quantiles: int = 32,
) -> FeedForwardNetwork:
  """Creates a value network."""
  if use_distributional_critic:
    value_module = DistributionalCritic(
        hidden_layer_sizes=list(hidden_layer_sizes),
        activation=activation,
        kernel_init=kernel_init,
        num_quantiles=num_quantiles,
    )
  else:
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [num_objectives],
        activation=activation,
        kernel_init=kernel_init,
    )

  def apply(processor_params, value_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    if use_distributional_critic:
      v_estimate, quantiles = value_module.apply(value_params, obs)
      return jnp.squeeze(v_estimate, axis=-1), quantiles
    else:
      v = value_module.apply(value_params, obs)
      # Drop the trailing axis only for a scalar critic; a bare squeeze would also
      # collapse leading axes that happen to have length 1.
      return jnp.squeeze(v, axis=-1) if num_objectives == 1 else v

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply
  )


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
) -> DesignHypernetNetworks:
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

    key_policy, key_value = jax.random.split(key)
    target_policy_params = policy_network.init(key_policy)
    target_value_params = value_network.init(key_value)

    obs_dim = networks._get_obs_state_size(observation_size, policy_obs_key)
    hypernetwork = make_design_hypernetwork(
        design_dim=design_dim,
        obs_dim=obs_dim,
        target_policy_dict=target_policy_params,
        target_value_dict=target_value_params,
        hypersize=hypersize,
        num_features=num_features,
        w_variance=w_variance,
    )

    return DesignHypernetNetworks(
        hypernetwork=hypernetwork,
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


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
        num_value_outputs=num_objectives,
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


def make_design_inference_fn(networks_: DesignHypernetNetworks):
    """Inference-fn factory keyed on the robot design.

    Returns ``inference_fn(params, design, deterministic=False) -> policy(obs, key)``,
    where ``params = (normalizer_params, hypernet_params)``. ``design`` may be a single
    design ``(design_dim,)`` or batched ``(num_envs, design_dim)``; in the batched case
    obs/params are vmapped over the leading env axis.
    """

    def design_inference_fn(
        params: types.Params, design: jax.Array, deterministic: bool = True
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

def make_value_fn(networks_: DesignHypernetNetworks):
    # Takes in the params and design and outputs a value function (obs -> value)
    def design_inference_fn(
        params: types.Params, design: jax.Array, tradeoff: jax.Array | None = None):
        normalizer_params, hypernet_params = params[0], params[1]
        value_network = networks_.value_network

        # Value params from the hypernetwork (ignore policy)
        condition = design if tradeoff is None else jnp.concatenate(
            [design, tradeoff], axis=-1
        )
        _, value_params = networks_.hypernetwork.apply(hypernet_params, condition)

        if len(design.shape) == 1:
            value_apply = value_network.apply
        else:
            value_apply = jax.vmap(value_network.apply, in_axes=(None, 0, 0))
        # Function that maps observation to value
        def value( observations: types.Observation):
            value = value_apply(normalizer_params, value_params, observations)
            return value

        return value

    return design_inference_fn
