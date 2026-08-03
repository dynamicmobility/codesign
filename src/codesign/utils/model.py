"""Design-construction helpers shared across training/eval.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import specs
from mujoco import mjx
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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


def maximin_designs(
    num_designs: int,
    low: float = 0.5,
    high: float = 2.0,
    dim: int = 1,
    n_restarts: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Spread ``num_designs`` designs to maximize the smallest gap between any two.

    Returns an array of shape ``(num_designs, dim)``.
    """
    if num_designs < 1:
        return np.empty((0, dim), np.float32)
    if num_designs == 1:
        return np.full((1, dim), 0.5 * (low + high), np.float32)
    if dim == 1:
        return np.linspace(low, high, num_designs, dtype=np.float32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    bounds = [(low, high)] * (num_designs * dim)

    def neg_min_gap(x):  # scipy minimizes, so negate
        return -np.min(pdist(x.reshape(num_designs, dim)))

    best, best_gap = None, -np.inf
    for _ in range(n_restarts):
        # Take res.x whether or not L-BFGS-B reports success -- on a nonsmooth objective it
        # often stops at a kink and still holds the best point of that restart.
        res = minimize(
            neg_min_gap,
            rng.uniform(low, high, num_designs * dim),
            bounds=bounds,
            method="L-BFGS-B",
        )
        if -res.fun > best_gap:
            best, best_gap = res.x, -res.fun

    return best.reshape(num_designs, dim).astype(np.float32)


def min_design_gap(designs: np.ndarray) -> float:
    """Smallest pairwise distance in a design set; ``inf`` for fewer than two designs."""
    designs = np.atleast_2d(np.asarray(designs, np.float64))
    if len(designs) < 2:
        return float("inf")
    return float(np.min(pdist(designs)))


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
