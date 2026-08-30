from typing import Any, Tuple, NamedTuple

import flax
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.types import Params
from brax.training.acme.types import NestedArray

from codesign.hyperdesigners import networks
from codesign.hyperdesigners.acting import DesignTransition


@flax.struct.dataclass
class DesignHypernetParams:
    """Trainable parameters: only the hypernetwork (target nets are not trained)."""

    hypernetwork: Params

class DesignPredictorTransition(NamedTuple):
    """One GRPO group per row: tradeoff ``w_t`` with ``G`` designs drawn from ``f(. | w_t)``."""

    tradeoff        : NestedArray # (n_tradeoffs, num_objectives)
    raw_design      : NestedArray # pre-tanh samples of f (n_tradeoffs, G, design_dim)
    value           : NestedArray # discounted scalarized returns (n_tradeoffs, G)
    design_log_prob : NestedArray # log f(raw_design | w) at sample time (n_tradeoffs, G)


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
        # Standardize each objective over the batch axes [T, B] only. The objective axis
        # M is not a batch axis: its entries carry different units, so they get their own
        # mean/std, which puts them on a common scale before the tradeoff mixes them.
        mean = advantages.mean(axis=(0, 1), keepdims=True)
        std = advantages.std(axis=(0, 1), keepdims=True)
        advantages = (advantages - mean) / (std + 1e-8)

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
    design_predictor_params: types.Params,
    data: DesignPredictorTransition,
    rng: jnp.ndarray,
    design_networks: networks.DesignPredictorHypernetNetworks,
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
