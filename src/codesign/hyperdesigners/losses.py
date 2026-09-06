"""Loss pieces shared by more than one hyperdesigner algo.

Each algo's own params container and loss live in the algo's module; the value-loss
shapes and the hypernetwork params container, which several of them reuse, stay here.
"""

import flax
import jax.numpy as jnp
from brax.training.types import Params


@flax.struct.dataclass
class DesignHypernetParams:
    """Trainable parameters: only the hypernetwork (target nets are not trained)."""

    hypernetwork: Params


def mse_loss(
    error: jnp.ndarray,
) -> jnp.ndarray:
    """PPO value loss, with the existing 0.5 value-loss coefficient."""
    loss = 0.5 * jnp.square(error)
    return 0.5 * jnp.mean(loss)

def huber_loss(
    error: jnp.ndarray,
    huber_delta: float = 1.0,
) -> jnp.ndarray:
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, huber_delta)
    linear = abs_error - quadratic
    loss = 0.5 * jnp.square(quadratic) + huber_delta * linear
    return 0.5 * jnp.mean(loss)
