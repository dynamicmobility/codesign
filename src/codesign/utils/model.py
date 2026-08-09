"""Design-construction helpers shared across training/eval.
"""

from collections.abc import Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import specs
from mujoco import mjx
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats.qmc import Sobol

from codesign.envs import CodesignBase

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
    codesign = config["env_config"]["codesign"]
    low = np.asarray(codesign["low"])
    high = np.asarray(codesign["high"])
    dim = len(codesign["low"])
    return np.linspace(low, high, num_envs).reshape(num_envs, dim).astype(np.float32)


def stack_models(models: Iterable[mjx.Model]) -> mjx.Model:
    """Stack same-topology ``mjx.Model``s along a new leading batch axis.

    Independently compiled models don't share a treedef (a few static fields differ),
    so we stack only the traced leaves and reuse the first model's treedef. Each model's
    leaves are copied to host arrays as it arrives, so an iterator of freshly compiled
    models keeps only one alive at a time.
    """
    cols, treedef = None, None
    for m in models:
        leaves, leaf_treedef = jax.tree_util.tree_flatten(m)
        if cols is None:
            cols, treedef = [[] for _ in leaves], leaf_treedef
        for col, leaf in zip(cols, leaves):
            col.append(np.asarray(leaf))
    return jax.tree_util.tree_unflatten(treedef, [jnp.stack(col) for col in cols])


def put_design_model(env: CodesignBase, design_row: np.ndarray) -> mjx.Model:
    """Build the env's ``mjx.Model`` for a single design row (host-side, ``mjx.put_model``'d)."""
    d = np.asarray(design_row, np.float32).reshape(-1)
    return mjx.put_model(env.generate_model(d))


def build_batched_model(env, designs: np.ndarray) -> mjx.Model:
    """Generate one ``mjx.Model`` per design row and stack them.

    Args:
        env: a ``CodesignBase`` env whose ``generate_model`` maps a design row -> a model.
        designs: array of shape ``(num_designs, design_dim)``.
    """
    return stack_models(put_design_model(env, d) for d in designs)


def sample_designs(
    rng: np.random.Generator,
    num_envs: int,
    low: float | list | np.ndarray,
    high: float | list | np.ndarray,
    dim: int = 1,
) -> np.ndarray:
    """Spread ``num_envs`` designs evenly through ``[low, high]^dim`` (host-side, numpy).

    Uses Sobol low-discrepancy sequence so the designs cover the hypercube far more
    uniformly than i.i.d. uniform samples for any num_envs number.
    """
    low = np.asarray(low)
    high = np.asarray(high)
    unit = Sobol(d=dim, seed=rng).random(num_envs)  # (num_envs, dim) in [0, 1)
    return (low + unit * (high - low)).astype(np.float32)


def maximin_designs(
    num_designs: int,
    low: float | list | np.ndarray = 0.5,
    high: float | list | np.ndarray = 2.0,
    dim: int = 1,
    n_restarts: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Spread ``num_designs`` designs to maximize the smallest gap between any two.

    ``low``/``high`` are per-dimension bounds broadcast to ``(dim,)``. The spread is
    solved in the unit cube and mapped onto the box, so each dimension weighs equally in
    the gap regardless of its physical range. Returns shape ``(num_designs, dim)``.
    """
    low = np.broadcast_to(np.asarray(low, np.float64), (dim,))
    high = np.broadcast_to(np.asarray(high, np.float64), (dim,))
    if num_designs < 1:
        return np.empty((0, dim), np.float32)
    if num_designs == 1:
        return (0.5 * (low + high)).reshape(1, dim).astype(np.float32)
    if dim == 1:
        return np.linspace(low[0], high[0], num_designs, dtype=np.float32).reshape(-1, 1)

    rng = np.random.default_rng(seed)

    def neg_min_gap(x):  # scipy minimizes, so negate
        return -np.min(pdist(x.reshape(num_designs, dim)))

    best, best_gap = None, -np.inf
    for _ in range(n_restarts):
        # Take res.x whether or not L-BFGS-B reports success -- on a nonsmooth objective it
        # often stops at a kink and still holds the best point of that restart.
        res = minimize(
            neg_min_gap,
            rng.random(num_designs * dim),
            bounds=[(0.0, 1.0)] * (num_designs * dim),
            method="L-BFGS-B",
        )
        if -res.fun > best_gap:
            best, best_gap = res.x, -res.fun

    return (low + best.reshape(num_designs, dim) * (high - low)).astype(np.float32)


def min_design_gap(designs: np.ndarray) -> float:
    """Smallest pairwise distance in a design set; ``inf`` for fewer than two designs."""
    designs = np.atleast_2d(np.asarray(designs, np.float64))
    if len(designs) < 2:
        return float("inf")
    return float(np.min(pdist(designs)))


def normalize_design(
    designs: jnp.ndarray, 
    low: float | np.ndarray = None, 
    high: float | np.ndarray = None,
    config: dict = None
) -> jnp.ndarray:
    """Normalize designs to ``[0, 1]`` using either explicit ``low``/``high`` or 
    a config dict. Defaults to config dict when provided."""
    if config is not None:
        codesign = config["env_config"]["codesign"]
        low = np.asarray(codesign["low"])
        high = np.asarray(codesign["high"])
    else:
        low = np.asarray(low)
        high = np.asarray(high)
    return (designs - low) / (high - low)

def unnormalize_design(
    designs: jnp.ndarray, 
    low: float | np.ndarray = None, 
    high: float | np.ndarray = None,
    config: dict = None
) -> jnp.ndarray:
    """Unnormalize designs from ``[0, 1]`` to ``[low, high]`` using either explicit 
    ``low``/``high`` or a config dict. Defaults to config dict when provided."""
    assert jnp.any((designs > 1) | (designs < 0)) == False
    if config is not None:
        codesign = config["env_config"]["codesign"]
        low = np.asarray(codesign["low"])
        high = np.asarray(codesign["high"])
    else:
        low = np.asarray(low)
        high = np.asarray(high)
    return low + designs * (high - low)

def observation_spec(observation_size):
    """Build a ``running_statistics`` spec from an env ``observation_size``.
    """

    def leaf(shp):
        dim = shp[-1] if isinstance(shp, (tuple, list)) else shp
        return specs.Array((int(dim),), jnp.dtype("float32"))

    if isinstance(observation_size, Mapping):
        return {k: leaf(v) for k, v in observation_size.items()}
    return leaf(observation_size)
