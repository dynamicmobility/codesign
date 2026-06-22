"""Design-construction helpers shared across training/eval.

Host-side helpers for turning robot designs into a stacked, batched ``mjx.Model``.
``MAICheetah.generate_model(d)`` recompiles the cheetah for a given design ``d`` (a
host-side ``spec.compile()`` — not jittable), so designs can only change at host
boundaries. We sample a batch of designs, build one ``mjx.Model`` per design, and stack
them along a leading batch axis so a single ``jax.vmap`` can roll out / train the whole
batch. The stacking pattern mirrors ``scripts/rollout_mai_cheetah.py``.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx


def uniform_design_sweep(config, num_envs: int) -> np.ndarray:
    """Uniform sweep of ``num_envs`` designs spanning the configured design range.

    Returns an array of shape ``(num_envs, design_dim)`` in ``[design_low, design_high]``.
    """
    design = config["learning_params"]["design_params"]
    low = float(design["design_low"])
    high = float(design["design_high"])
    dim = int(design["design_dim"])
    return np.linspace(low, high, num_envs).reshape(num_envs, dim).astype(np.float32)


def stack_models(models: list[mjx.Model]) -> mjx.Model:
    """Stack same-topology ``mjx.Model``s along a new leading batch axis.

    Independently compiled models don't share a treedef (a few static fields differ),
    so we stack only the traced leaves and reuse the first model's treedef.
    """
    leaves = [jax.tree_util.tree_leaves(m) for m in models]
    _, treedef = jax.tree_util.tree_flatten(models[0])
    return jax.tree_util.tree_unflatten(
        treedef, [jnp.stack(col) for col in zip(*leaves)]
    )


def build_batched_model(
    generate_model_fn: Callable[[np.ndarray], mjx.Model],
    designs: np.ndarray,
) -> mjx.Model:
    """Generate one ``mjx.Model`` per design row and stack them.

    Args:
        generate_model_fn: maps a single design row -> an ``mjx.Model`` (already
            ``mjx.put_model``'d).
        designs: array of shape ``(num_envs, design_dim)``.
    """
    models = [generate_model_fn(np.asarray(d)) for d in designs]
    return stack_models(models)


def sample_designs(
    rng: np.random.Generator,
    num_envs: int,
    low: float = 0.5,
    high: float = 2.0,
    dim: int = 1,
) -> np.ndarray:
    """Sample ``num_envs`` designs uniformly in ``[low, high]^dim`` (host-side, numpy)."""
    return rng.uniform(low, high, size=(num_envs, dim)).astype(np.float32)


def normalize_design(
    designs: jnp.ndarray, low: float = 0.5, high: float = 2.0
) -> jnp.ndarray:
    """Map raw designs in ``[low, high]`` to ``[0, 1]`` for the hypernetwork input."""
    return (designs - low) / (high - low)
