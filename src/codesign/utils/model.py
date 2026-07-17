"""Design-construction helpers shared across training/eval.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import specs
from mujoco import mjx
from scipy.stats.qmc import Sobol

def total_mass(model) -> float:
    """Total mass of the model underlying a ``CodesignBase`` env.

    Sums ``body_mass`` over every body in the env's compiled ``mj_model`` (the
    worldbody contributes 0), giving the model's total mass in kilograms.
    """
    return float(np.sum(model.body_mass))


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


def put_design_model(env, design_row: np.ndarray) -> mjx.Model:
    """Build the env's ``mjx.Model`` for a single design row (host-side, ``mjx.put_model``'d)."""
    d = float(np.asarray(design_row).reshape(-1)[0])
    return mjx.put_model(env.generate_model(d))


def build_batched_model(env, designs: np.ndarray) -> mjx.Model:
    """Generate one ``mjx.Model`` per design row and stack them.

    Args:
        env: a ``CodesignBase`` env whose ``generate_model`` maps a design row -> a model.
        designs: array of shape ``(num_envs, design_dim)``.
    """
    models = [put_design_model(env, d) for d in designs]
    return stack_models(models)


def sample_designs(
    rng: np.random.Generator,
    num_envs: int,
    low: float = 0.5,
    high: float = 2.0,
    dim: int = 1,
) -> np.ndarray:
    """Spread ``num_envs`` designs evenly through ``[low, high]^dim`` (host-side, numpy).

    Uses Sobol low-discrepancy sequence so the designs cover the hypercube far more
    uniformly than i.i.d. uniform samples for any num_envs number.
    """
    unit = Sobol(d=dim, seed=rng).random(num_envs)  # (num_envs, dim) in [0, 1)
    return (low + unit * (high - low)).astype(np.float32)


def normalize_design(
    designs: jnp.ndarray, low: float = None, high: float = None, config: dict = None
) -> jnp.ndarray:
    """Normalize designs to ``[0, 1]`` using either explicit ``low``/``high`` or 
    a config dict. Defaults to config dict when provided."""
    if config is not None:
        design_params = config["learning_params"]["design_params"]
        low = float(design_params["design_low"])
        high = float(design_params["design_high"])
    return (designs - low) / (high - low)

def observation_spec(observation_size):
    """Build a ``running_statistics`` spec from an env ``observation_size``.
    """

    def leaf(shp):
        dim = shp[-1] if isinstance(shp, (tuple, list)) else shp
        return specs.Array((int(dim),), jnp.dtype("float32"))

    if isinstance(observation_size, Mapping):
        return {k: leaf(v) for k, v in observation_size.items()}
    return leaf(observation_size)
