"""The design x tradeoff grid shared by training, evaluation, and saved datasets.
"""

import dataclasses
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from codesign.envs.codesign_base import CodesignBase, MOCodesignBase
from codesign.utils.model import (
    build_batched_model,
    normalize_design,
    sample_designs,
    unnormalize_design,
)


def sample_tradeoffs(
    rng: np.random.Generator,
    num_tradeoffs: int,
    num_objectives: int,
    sampling: str = "dense",
    alpha: float = 1.0,
) -> np.ndarray:
    """Sample ``num_tradeoffs`` simplex tradeoffs (host-side numpy):
      * ``dense`` — ``num_tradeoffs`` Dirichlet(alpha) draws;
      * ``sparse-heavytail`` — ``num_tradeoffs - num_objectives`` Dirichlet draws plus the
        ``num_objectives`` axis-aligned (one-hot) extreme tradeoffs (e.g. ``num_tradeoffs=8``,
        ``num_objectives=3`` -> 5 simplex draws + 3 one-hot corners);
      * ``single-avg`` — every tradeoff is the uniform ``1/M``.
    Returns an array of shape ``(num_tradeoffs, num_objectives)``.
    """

    if sampling == "dense":
        w = rng.dirichlet(np.ones(num_objectives) * alpha, size=num_tradeoffs)
    elif sampling == "sparse-heavytail":
        n_dir = max(num_tradeoffs - num_objectives, 0)
        dir_w = rng.dirichlet(np.ones(num_objectives) * alpha, size=n_dir)
        w = np.concatenate([dir_w, np.eye(num_objectives)], axis=0)
        w = w[:num_tradeoffs]
    elif sampling == "single-avg":
        w = np.full((num_tradeoffs, num_objectives), 1.0 / num_objectives)
    else:
        raise ValueError(f"Sampling type {sampling} not implemented")
    return w.astype(np.float32)


def _generator(seed: int | jax.Array | np.random.Generator) -> np.random.Generator:
    """A ``Generator`` from an int seed, or a ``Generator`` passed straight through.

    ``np.random.default_rng`` returns a ``Generator`` unaltered, so a caller may hand over
    either a fresh seed or a live stream that keeps advancing across resamples.
    """
    return np.random.default_rng(int(seed) if isinstance(seed, jax.Array) else seed)


def _box_designs(seed, env: CodesignBase, n_designs: int, limits = None) -> np.ndarray:
    """Space-filling sample of ``n_designs`` over the env's design box.

    ``env.design_limits`` is ``(2, design_dim)`` -- lows then highs. Returns shape
    ``(n_designs, design_dim)`` in physical units.
    """
    if(limits == None):
        low, high = np.asarray(env.design_limits)
    else:
        low, high = limits
    return sample_designs(_generator(seed), n_designs, low=low, high=high, dim=len(low))



@dataclasses.dataclass
class Grid:
    """``M`` designs x ``K`` tradeoffs, each cell rolled out ``per_cell`` times.

    Every sampling stage in the repository is one of these:
      * single objective -- ``(M, 1, C, 1)``: one trivial tradeoff, the weight 1.0 on the
        env's already-scalar reward;
      * crossed -- ``designs[m, k] == designs[m, 0]``: every design meets every tradeoff;
      * paired -- ``designs[m, k] ~ f(. | tradeoffs[0, k])``: cell ``(m, k)`` is the
        ``m``-th member of tradeoff ``k``'s group, so the group size *is* ``M``.
    Storing both arrays fully broadcast to ``(M, K, ...)`` is what makes those one type:
    which design met which tradeoff is read off the cell, not off the class.

    ``rewards is None`` marks a grid that has been sampled but not yet rolled out; the
    two are otherwise the same object, so a rollout only fills in a field.
    """

    designs    : np.ndarray                # (M, K, design_dim), physical units
    tradeoffs  : np.ndarray                # (M, K, n_r), simplex weights
    per_cell   : int = 1                   # C, rollout repetitions per cell
    rewards    : np.ndarray | None = None  # (M, K, C, n_r), accumulated return
    objectives : list | None = None        # per-objective names
    # per-step trajectories, (M, K, C, n_steps, ...); per-cell values omit the time axis
    data       : dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    _DATA_PREFIX = "data/"  # npz namespace keeping ``data`` apart from the grid arrays

    # ------------------------------------------------------------------ constructors

    @classmethod
    def crossed(cls, designs, tradeoffs, per_cell: int = 1, **kwargs) -> "Grid":
        """Cross ``(M, design_dim)`` designs with ``(K, n_r)`` tradeoffs (repeat 
        design for each tradeoff and vice versa)"""
        designs = np.atleast_2d(np.asarray(designs, np.float32))
        tradeoffs = np.atleast_2d(np.asarray(tradeoffs, np.float32))
        cells = (len(designs), len(tradeoffs))
        return cls(
            designs   = np.broadcast_to(
                designs[:, None], cells + designs.shape[-1:]
            ).copy(),
            tradeoffs = np.broadcast_to(
                tradeoffs[None], cells + tradeoffs.shape[-1:]
            ).copy(),
            per_cell  = per_cell,
            **kwargs,
        )

    @classmethod
    def paired(cls, designs, tradeoffs, per_cell: int = 1, **kwargs) -> "Grid":
        """Pair ``(M, K, design_dim)`` designs with the ``(K, n_r)`` tradeoff of their column.
        """
        designs = np.asarray(designs, np.float32)
        tradeoffs = np.atleast_2d(np.asarray(tradeoffs, np.float32))
        return cls(
            designs   = designs,
            tradeoffs = np.broadcast_to(
                tradeoffs[None], designs.shape[:2] + tradeoffs.shape[-1:]
            ).copy(),
            per_cell  = per_cell,
            **kwargs,
        )

    @classmethod
    def from_design_sample(
        cls, env: CodesignBase, seed, n_designs: int, per_cell: int = 1, limits = None,
    ) -> "Grid":
        """Space-filling (sobol) designs against the single trivial tradeoff, ``(M, 1, C, 1)``.
        """
        return cls.crossed(
            _box_designs(seed, env, n_designs, limits = None), np.ones((1, 1), np.float32), per_cell
        )

    @classmethod
    def from_uniform_sample(
        cls,
        env: MOCodesignBase,
        seed,
        n_tradeoffs: int,
        n_designs: int,
        per_cell: int = 1,
        sampling: str = "sparse-heavytail",
        alpha: float = 1.0,
    ) -> "Grid":
        """Space-filling designs over the env's design box crossed with sampled tradeoffs.
        """
        rng = _generator(seed)
        tradeoffs = sample_tradeoffs(
            rng, n_tradeoffs, len(env.objectives), sampling=sampling, alpha=alpha
        )
        return cls.crossed(
            _box_designs(rng, env, n_designs), tradeoffs, per_cell,
            objectives=env.objectives,
        )

    @classmethod
    def from_simplex_corners(
        cls, env: MOCodesignBase, seed, n_designs: int, per_cell: int = 1
    ) -> "Grid":
        """Space-filling designs crossed with the one-hot corners of the simplex."""
        return cls.crossed(
            _box_designs(seed, env, n_designs),
            np.eye(len(env.objectives), dtype=np.float32),
            per_cell,
            objectives=env.objectives,
        )

    @classmethod
    def from_2d_tradeoffs(
        cls,
        env: MOCodesignBase,
        seed,
        n_tradeoffs_per_pair: int,
        n_designs: int,
        per_cell: int = 1,
    ) -> "Grid":
        """Samples tradeoffs purely in two objectives.
        """
        m = len(env.objectives)
        if m < 2:
            raise ValueError(f"2D tradeoffs need at least 2 objectives; env has {m}.")
        x = np.linspace(0.0, 1.0, n_tradeoffs_per_pair, dtype=np.float32)
        blocks = []
        for i, j in itertools.combinations(range(m), 2):
            block = np.zeros((n_tradeoffs_per_pair, m), dtype=np.float32)
            block[:, i], block[:, j] = x, 1.0 - x
            blocks.append(block)
        return cls.crossed(
            _box_designs(seed, env, n_designs),
            np.concatenate(blocks, axis=0),
            per_cell,
            objectives=env.objectives,
        )

    @classmethod
    def from_predictor(
        cls,
        env: MOCodesignBase,
        design_predictor_inference_fn,
        design_params,
        seed,
        n_tradeoffs: int,
        n_designs: int = 1,
        per_cell: int = 1,
        sampling: str = "sparse-heavytail",
        alpha: float = 1.0,
        tradeoffs=None,
        key=None,
        deterministic: bool | None = None,
    ) -> tuple["Grid", dict | None]:
        """Designs drawn from the trained predictor ``f(d | w)``, one group per tradeoff.
        """
        rng = _generator(seed)
        if tradeoffs is None:
            tradeoffs = sample_tradeoffs(
                rng, n_tradeoffs, len(env.objectives), sampling=sampling, alpha=alpha
            )
        tradeoffs = np.atleast_2d(np.asarray(tradeoffs, np.float32))
        if deterministic is None:
            deterministic = n_designs == 1
        if key is None:
            key = jax.random.PRNGKey(int(rng.integers(1 << 31)))

        # One predictor evaluation per (tradeoff, group member), tradeoff-major.
        designs_norm, extras = design_predictor_inference_fn(
            design_params,
            jnp.repeat(jnp.asarray(tradeoffs), n_designs, axis=0),
            deterministic = deterministic,
            key_sample    = key,
        )
        groups = lambda x: jnp.reshape(x, (n_tradeoffs, n_designs, *x.shape[1:]))
        low, high = np.asarray(env.design_limits)
        # Designs come out normalized to [0, 1]; the model generator wants physical units.
        designs = unnormalize_design(groups(designs_norm), low, high)
        grid = cls.paired(
            np.swapaxes(np.asarray(designs), 0, 1), tradeoffs, per_cell,
            objectives=env.objectives,
        )
        # A deterministic query has no sample to score, so it returns no extras.
        return grid, ({k: groups(v) for k, v in extras.items()} if extras else None)

    # ------------------------------------------------------------------ shape

    @property
    def n_designs(self) -> int:
        """``M``: designs per tradeoff, which for a paired grid is the group size."""
        return self.designs.shape[0]

    @property
    def n_tradeoffs(self) -> int:
        """``K``."""
        return self.designs.shape[1]

    @property
    def n_r(self) -> int:
        """``n_r``: the number of objectives each reward vector carries."""
        return self.tradeoffs.shape[-1]

    @property
    def design_dim(self) -> int:
        return self.designs.shape[-1]

    @property
    def cell_shape(self) -> tuple[int, int, int]:
        """``(M, K, C)`` -- the axes every per-env quantity unflattens onto."""
        return (self.n_designs, self.n_tradeoffs, self.per_cell)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(M, K, C, n_r)`` -- the shape :attr:`rewards` takes once rolled out."""
        return self.cell_shape + (self.n_r,)

    @property
    def num_envs(self) -> int:
        return self.n_designs * self.n_tradeoffs * self.per_cell

    @property
    def unique_tradeoffs(self) -> np.ndarray:
        """The ``K`` tradeoffs, ``(K, n_r)``; every design column carries the same ones."""
        return self.tradeoffs[0]

    @property
    def mean_rewards(self) -> np.ndarray:
        """Rewards averaged over the per-cell repetition axis, ``(M, K, n_r)``."""
        return self.rewards.mean(axis=2)

    @property
    def keys(self) -> list[str]:
        """Names of the recorded per-step quantities."""
        return sorted(self.data)

    # ------------------------------------------------------------------ env axis

    def flatten(self) -> tuple[np.ndarray, np.ndarray]:
        """Designs and tradeoffs on the flat env axis ``((m * K) + k) * per_cell + c``.

        Returns ``(designs_full, tradeoffs_full)``, each with leading axis :attr:`num_envs`.
        """
        tile = lambda x: np.repeat(x.reshape(-1, x.shape[-1]), self.per_cell, axis=0)
        return tile(self.designs), tile(self.tradeoffs)

    def unflatten(self, x):
        """Reshape a flat env-axis array to ``(M, K, C, ...)``, leaving its backend alone."""
        return x.reshape(*self.cell_shape, *x.shape[1:])

    def build_models(
        self, env: CodesignBase, tiled: bool = True, workers: int = 1, like=None
    ) -> mjx.Model:
        """One ``mjx.Model`` per cell, stacked over ``(M, K)`` or the flat env axis.

        Only unique designs are compiled. Multi-threading available through workers.
        """
        rows = self.designs.reshape(-1, self.design_dim)
        unique, inverse = np.unique(rows, axis=0, return_inverse=True)
        stacked = build_batched_model(env, unique, workers=workers)
        index = np.ravel(inverse)
        index = np.repeat(index, self.per_cell) if tiled else index
        models = jax.tree_util.tree_map(lambda x: x[index], stacked)
        if not tiled:
            models = jax.tree_util.tree_map(
                lambda x: x.reshape(self.n_designs, self.n_tradeoffs, *x.shape[1:]),
                models,
            )
        if like is None:
            return models
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(like), jax.tree_util.tree_leaves(models)
        )

    def env_inputs(self, env: CodesignBase, workers: int = 1, like=None):
        """The flat env-axis rollout inputs of this grid.

        Returns ``(batched_model, designs, tradeoffs)``, each with leading axis
        :attr:`num_envs`. ``designs`` are normalized to ``[0, 1]`` because that is what the
        hypernetwork is conditioned on, while the grid itself holds physical units.
        ``workers`` and ``like`` are :meth:`build_models`'.
        """
        designs, tradeoffs = self.flatten()
        low, high = np.asarray(env.design_limits)
        return (
            self.build_models(env, tiled=True, workers=workers, like=like),
            normalize_design(jnp.asarray(designs), low, high),
            jnp.asarray(tradeoffs),
        )

    # ------------------------------------------------------------------ batching

    def _batch(self, axis: int, n_batches: int, rng=None) -> list["Grid"]:
        """Split ``axis`` (0 designs, 1 tradeoffs) into ``n_batches`` equally sized grids.

        ``rng`` permutes the axis first, so each batch mixes entries from across the grid;
        without it the split is contiguous, keeping neighbouring grid entries together --
        which is what a within-design batch needs.
        """
        n = self.designs.shape[axis]
        if n % n_batches:
            raise ValueError(
                f"axis {axis} has length {n}, which {n_batches} batches do not divide"
            )
        order = _generator(rng).permutation(n) if rng is not None else np.arange(n)
        take = lambda x, idx: None if x is None else np.take(x, idx, axis=axis)
        return [
            dataclasses.replace(
                self,
                designs   = take(self.designs, idx),
                tradeoffs = take(self.tradeoffs, idx),
                rewards   = take(self.rewards, idx),
                data      = {k: take(v, idx) for k, v in self.data.items()},
            )
            for idx in np.split(order, n_batches)
        ]

    def batch_by_design(self, n_batches: int, rng=None) -> list["Grid"]:
        """``n_batches`` grids of ``M / n_batches`` designs each; see :meth:`_batch`."""
        return self._batch(0, n_batches, rng)

    def batch_by_tradeoff(self, n_batches: int, rng=None) -> list["Grid"]:
        """``n_batches`` grids of ``K / n_batches`` tradeoffs each; see :meth:`_batch`."""
        return self._batch(1, n_batches, rng)

    # ------------------------------------------------------------------ i/o

    def save(self, path: str | Path) -> None:
        """Write the grid to ``path`` as an npz that :meth:`load` reads back."""
        arrays = dict(
            designs=self.designs, tradeoffs=self.tradeoffs, per_cell=self.per_cell
        )
        if self.rewards is not None:
            arrays["rewards"] = self.rewards
        if self.objectives is not None:
            arrays["objectives"] = np.asarray(self.objectives, dtype=object)
        arrays.update({self._DATA_PREFIX + k: v for k, v in self.data.items()})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Grid":
        npz = np.load(path, allow_pickle=True)
        cut = len(cls._DATA_PREFIX)
        return cls(
            designs    = npz["designs"],
            tradeoffs  = npz["tradeoffs"],
            per_cell   = int(npz["per_cell"]),
            rewards    = npz["rewards"] if "rewards" in npz else None,
            objectives = npz["objectives"].tolist() if "objectives" in npz else None,
            data       = {
                k[cut:]: npz[k] for k in npz.files if k.startswith(cls._DATA_PREFIX)
            },
        )
