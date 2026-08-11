from typing import Any, Tuple, NamedTuple

import flax
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.types import Params
from brax.training.acme.types import NestedArray

from codesign.hyperdesigners import networks
from codesign.hyperdesigners.acting import DesignTransition, MODesignTransition


@flax.struct.dataclass
class DesignHypernetParams:
    """Trainable parameters: only the hypernetwork (target nets are not trained)."""

    hypernetwork: Params

class DesignPredictorTransition(NamedTuple):
    # G is the number of trials in a group (based on a single tradeoff)
    tradeoff        : NestedArray # One tradeoff (n_tradeoff, )
    design          : NestedArray # design vectors from repeated sampling of f (G, design_dim)
    value           : NestedArray # values from scalarized rollout with discounted returns (G, ) 
    design_log_prob : NestedArray # log probabilities that a particular design was drawn (G, )


def compute_design_hypernet_loss(
    params: DesignHypernetParams,
    normalizer_params: Any,
    data: DesignTransition,
    rng: jnp.ndarray,
    design_networks: networks.DesignHypernetNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
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
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

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


def compute_mo_design_hypernet_loss(
    params: DesignHypernetParams,
    normalizer_params: Any,
    data: MODesignTransition,
    rng: jnp.ndarray,
    design_networks: networks.DesignHypernetNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes the clipped-PPO loss for the multi-objective design hypernetwork ``H(d, w)``.

    Args:
        params: trainable hypernetwork params.
        normalizer_params: observation normalizer params.
        data: ``MODesignTransition`` with leading dims ``[B, T]``. ``reward``/``tradeoff``
            carry a trailing objective axis of size ``M``. Requires
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

    # Per-env policy/value params from the hypernetwork. Design and tradeoff are constant
    # over time, so use their value at the first timestep.
    cond = jnp.concatenate(
        [data.design[:, 0], data.tradeoff[:, 0]], axis=-1
    )
    policy_params, value_params = design_networks.hypernetwork.apply(
        params.hypernetwork, cond
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

    # Scalarize the per-objective reward by the per-step tradeoff: [T, B, M] -> [T, B].
    rewards = jnp.sum(data.tradeoff * data.reward, axis=2) * reward_scaling

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
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

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


def compute_grpo_loss(
    design_networks: networks.DesignPredictorHypernetNetworks,
    design_predictor_params: types.Params,
    data: DesignPredictorTransition,
    rng,
    entropy_cost: float = 1e-4,
    clipping_epsilon: float = 0.3,
    ):
    # Run design predictor k times for each design
    # each group is size k

    parametric_design_distribution = design_networks.parametric_design_distribution
    design_predictor_inference_fn = networks.make_design_predictor_inference_fn(design_networks)

    # Tile tradeoff G times to sample G target designs
    target_design_logits = design_predictor_inference_fn(design_predictor_params, data.tradeoff, deterministic=False, key_sample=rng)

    target_design_log_prob = parametric_design_distribution.log_prob(
        target_design_logits, data.design
    )
    # Policy Ratio rho is the difference of log probs from the original and updated distribution 
    rho_s = jnp.exp(target_design_log_prob - data.design_log_prob)

    # Group-relative advantage calculation:

    r_mean = jnp.mean(data.value)
    r_std = jnp.std(data.value)

    advantages = (data.value-r_mean)/r_std

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))


    # Entropy bonus.
    entropy = jnp.mean(parametric_design_distribution.entropy(target_design_logits, rng))
    entropy_loss = entropy_cost * -entropy
    
    total_loss = policy_loss + entropy_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "entropy_loss": entropy_loss,
    }
