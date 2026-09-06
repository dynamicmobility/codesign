"""Network pieces shared by more than one hyperdesigner algo.

Each algo's own bundle, builders and inference fns live in the algo's module; only what
several of them reuse stays here.
"""

import dataclasses
import inspect
from typing import Any, Callable, Literal, Sequence

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.networks import ActivationFn, DistributionalCritic, Initializer, FeedForwardNetwork, MLP, Mapping
from brax.training.networks import normalizer_select, _get_obs_state_size
from flax import linen


@dataclasses.dataclass
class FeedForwardHypernetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]
    # ``(params, design) -> (batch, num_features)``, the row that multiplies W.
    features: Callable[..., Any] = None


@flax.struct.dataclass
class DesignHypernetNetworks:
    hypernetwork: FeedForwardHypernetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def resolve_kernel_initializer(initializer: str | Initializer) -> Initializer:
    """A ready ``(key, shape, dtype)`` initializer from a name in brax's registry.

    The registry mixes factories (``he_uniform``) with initializers that are already in that
    form (``zeros``); only a factory needs calling, and one is told apart by taking no
    ``shape`` argument. An initializer handed over directly comes back unchanged.

    Names, rather than initializers, are what the builders take: brax's checkpoint writer
    JSON-encodes a network factory's keyword arguments, and a function there is written out
    as an unusable ``"function init"``.
    """
    if not isinstance(initializer, str):
        return initializer
    initializer = networks.KERNEL_INITIALIZER[initializer]
    if "shape" in inspect.signature(initializer).parameters:
        return initializer
    return initializer()


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



def make_target_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    key: jax.Array,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (64,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (64,) * 2,
    activation: ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    num_value_outputs: int = 1,
    weight_initializer: str | Initializer = "kaiming_uniform",
):
    """The target policy/value MLPs whose weights a hypernetwork generates.

    Returns ``(parametric_action_distribution, policy_network, value_network,
    target_policy_params, target_value_params, obs_dim)``, where the two param trees are one
    draw of the target nets' own initialization and ``obs_dim`` is the policy's flat
    observation width.
    """
    weight_initializer = resolve_kernel_initializer(weight_initializer)
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
        kernel_init=weight_initializer,
    )

    value_network = make_vector_value_network(
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        num_objectives=num_value_outputs,
        activation=activation,
        obs_key=value_obs_key,
        kernel_init=weight_initializer,
    )

    key_policy, key_value = jax.random.split(key)
    return (
        parametric_action_distribution,
        policy_network,
        value_network,
        policy_network.init(key_policy),
        value_network.init(key_value),
        _get_obs_state_size(observation_size, policy_obs_key),
    )


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
