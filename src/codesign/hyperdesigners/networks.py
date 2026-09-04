"""Network pieces shared by more than one hyperdesigner algo.

Each algo's own bundle, builders and inference fns live in the algo's module; only what
several of them reuse stays here.
"""

import dataclasses
from typing import Any, Callable, Sequence

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


@flax.struct.dataclass
class DesignHypernetNetworks:
    hypernetwork: FeedForwardHypernetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


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
