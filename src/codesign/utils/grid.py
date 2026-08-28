"""Design x tradeoff grid structures shared by training and evaluation.
"""

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import jax
from mujoco import mjx

from codesign.utils.model import build_batched_model, sample_designs, unnormalize_design
from codesign.envs.codesign_base import CodesignBase, MOCodesignBase

def sample_tradeoffs(
    rng: np.random.Generator,
    it: int,
    num_tradeoffs: int,
    num_objectives: int,
    sampling: str = "dense",
    alpha: float = 1.0,
) -> np.ndarray:
    """Sample ``num_tradeoffs`` simplex tradeoffs (host-side numpy), MORLAX-style.

    ``num_tradeoffs`` is the sole driver of how many distinct tradeoffs are produced.
    Ports the sampling styles of ``morlax.sample_preferences``:
      * ``dense`` — ``num_tradeoffs`` Dirichlet(alpha) draws;
      * ``sparse-heavytail`` — ``num_tradeoffs - num_objectives`` Dirichlet draws plus the
        ``num_objectives`` axis-aligned (one-hot) extreme tradeoffs (e.g. ``num_tradeoffs=8``,
        ``num_objectives=3`` -> 5 simplex draws + 3 one-hot corners);
      * ``single-avg`` — every tradeoff is the uniform ``1/M``.
    During warmup (``it < round(warmup_frac * num_warmup_ref)``) all tradeoffs are uniform.
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


def _box_designs(rng: np.random.Generator, env: CodesignBase, n_designs: int) -> np.ndarray:
    """Space-filling sample of ``n_designs`` over the env's design box.

    Returns shape ``(n_designs, design_dim)`` in physical units.
    """
    limits = np.asarray(env.design_limits)
    return sample_designs(
        rng, n_designs, low=limits[0], high=limits[1], dim=limits.shape[1]
    )


@dataclasses.dataclass
class DesignTradeoffSampleGrid:
    """A sampled design x tradeoff grid and its flat env-axis ordering."""

    designs: np.ndarray      # (n_designs, design_dim)
    tradeoffs: np.ndarray    # (n_tradeoffs, num_objectives)
    per_cell: int = 1        # rollout repetitions per (design, tradeoff) cell
    
    @classmethod
    def from_uniform_sample(
        cls,
        env: CodesignBase,
        seed: int,
        n_tradeoffs,
        n_designs,
        per_cell: int = 1,
        sampling: str = "sparse-heavytail",
    ) -> "DesignTradeoffSampleGrid":
        """Space-filling designs over the env's design box crossed with sampled tradeoffs.

        ``sampling`` is passed to :func:`sample_tradeoffs`; under the default
        ``sparse-heavytail`` asking for ``n_tradeoffs == num_objectives`` returns exactly
        the one-hot corners of the simplex (the single-objective extremes).
        """
        rng = np.random.default_rng(seed)
        tradeoffs = sample_tradeoffs(
            rng, 0, n_tradeoffs, len(env.objectives), sampling=sampling
        )
        return cls(
            designs=_box_designs(rng, env, n_designs),
            tradeoffs=tradeoffs,
            per_cell=per_cell,
        )
        
    @classmethod
    def from_simplex_corners(
        cls,
        env: MOCodesignBase,
        seed: int,
        n_designs,
        per_cell: int = 1,
    ) -> "DesignTradeoffSampleGrid":
        """Space-filling designs crossed with the one-hot corners of the simplex.
        """
        m = len(env.objectives)
        return cls(
            designs=_box_designs(np.random.default_rng(seed), env, n_designs),
            tradeoffs=np.eye(m, dtype=np.float32),
            per_cell=per_cell,
        )

    @classmethod
    def from_2d_tradeoffs(
        cls,
        env: MOCodesignBase,
        seed: int,
        n_tradeoffs_per_pair,
        n_designs,
        per_cell: int = 1,
    ) -> "DesignTradeoffSampleGrid":
        """Samples tradeoffs purely in two objectives. For instance when
        m = 3, samples [x, y, 0], [x, 0, y], [0, x, y]. Useful for when plotting
        2D paretos from a 3D mo design hypernetwork.
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
        return cls(
            designs=_box_designs(np.random.default_rng(seed), env, n_designs),
            tradeoffs=np.concatenate(blocks, axis=0),
            per_cell=per_cell,
        )


    @property
    def n_designs(self) -> int:
        return self.designs.shape[0]

    @property
    def n_tradeoffs(self) -> int:
        return self.tradeoffs.shape[0]
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Leading grid axes of a rollout over this grid."""
        return (self.n_designs, self.n_tradeoffs, self.per_cell)

    @property
    def num_envs(self) -> int:
        return self.n_designs * self.n_tradeoffs * self.per_cell

    def build_models(self, env: CodesignBase, tiled: bool, workers: int = 1) -> mjx.Model:
        """Stack one model per design (``tiled=False``) or per flat env (``tiled=True``).

        ``workers`` threads the per-design compiles; see :func:`build_batched_model`.
        """
        stacked = build_batched_model(env, self.designs, workers=workers)
        if not tiled:
            return stacked
        reps = self.n_tradeoffs * self.per_cell
        return jax.tree_util.tree_map(
            lambda x: jax.numpy.repeat(x, reps, axis=0), stacked
        )

    def flatten(self) -> tuple[np.ndarray, np.ndarray]:
        """Tile designs/tradeoffs to the flat env axis.

        Returns ``(designs_full, tradeoffs_full)``, each with leading axis ``num_envs``.
        """
        reps = self.n_tradeoffs * self.per_cell
        designs_full = np.repeat(self.designs, reps, axis=0)
        tradeoffs_full = np.tile(
            np.repeat(self.tradeoffs, self.per_cell, axis=0), (self.n_designs, 1)
        )
        return designs_full, tradeoffs_full

    def unflatten(self, x) -> np.ndarray:
        """Reshape a flat env-axis array to ``(n_designs, n_tradeoffs, per_cell, ...)``."""
        x = np.asarray(x)
        return x.reshape(self.n_designs, self.n_tradeoffs, self.per_cell, *x.shape[1:])

    def to_grid_axes(self, x) -> np.ndarray:
        """Map an array with :attr:`shape`'s leading axes to ``(design, tradeoff, rep, ...)``.

        Already the rollout's own axis order here, so this is a plain conversion.
        """
        return np.asarray(x)


@dataclasses.dataclass
class DesignPredictorSampleGrid:
    """Designs paired with the tradeoff they were drawn for, and its flat env ordering.

    Unlike :class:`DesignTradeoffSampleGrid`, designs are not crossed with tradeoffs:
    ``designs[t, g]`` was sampled from ``f(. | tradeoffs[t])`` and belongs only to that
    tradeoff. The ``group_size`` axis is the GRPO group.
    """

    designs: np.ndarray      # (n_tradeoffs, group_size, design_dim), physical units
    tradeoffs: np.ndarray    # (n_tradeoffs, num_objectives)
    per_cell: int = 1        # rollout repetitions per (tradeoff, design) cell

    @classmethod
    def from_predictor(
        cls,
        env: CodesignBase,
        design_predictor_inference_fn,
        design_params,
        seed: int,
        n_tradeoffs: int,
        group_size: int = 1,
        per_cell: int = 1,
        sampling: str = "sparse-heavytail",
    ) -> "DesignPredictorSampleGrid":
        """Designs drawn from the trained predictor ``f(d | w)``, one group per tradeoff.

        ``group_size == 1`` takes the predictor's mode (its predicted optimum for that
        tradeoff); with more, designs are sampled from ``f`` so the group spreads around
        it. ``sampling`` is passed to :func:`sample_tradeoffs`.
        """
        tradeoffs = sample_tradeoffs(
            np.random.default_rng(seed), 0, n_tradeoffs, len(env.objectives),
            sampling=sampling,
        )
        # One predictor evaluation per (tradeoff, group member).
        tradeoffs_tiled = np.repeat(tradeoffs, group_size, axis=0)
        designs_norm, _ = design_predictor_inference_fn(
            design_params,
            jax.numpy.asarray(tradeoffs_tiled),
            deterministic = group_size == 1,
            key_sample    = jax.random.PRNGKey(seed + 1),
        )
        # Designs come out normalized to [0, 1]; the model generator wants physical units.
        limits = np.asarray(env.design_limits)
        designs = unnormalize_design(
            designs_norm.reshape(n_tradeoffs, group_size, -1), limits[0], limits[1]
        )
        return cls(
            designs=np.asarray(designs),
            tradeoffs=tradeoffs,
            per_cell=per_cell,
        )

    @property
    def n_tradeoffs(self) -> int:
        return self.tradeoffs.shape[0]

    @property
    def n_designs(self) -> int:
        """Designs per tradeoff -- the design axis of a rollout is the group axis."""
        return self.group_size

    @property
    def group_size(self) -> int:
        return self.designs.shape[1]

    @property
    def num_envs(self) -> int:
        return self.n_tradeoffs * self.group_size * self.per_cell

    @property
    def shape(self) -> tuple[int, int, int]:
        """Leading grid axes of a rollout over this grid, group-major."""
        return (self.n_tradeoffs, self.group_size, self.per_cell)

    def build_models(
        self, env: CodesignBase, tiled: bool = True, workers: int = 1
    ) -> mjx.Model:
        """One model per ``(tradeoff, design)`` pair.

        ``tiled=True`` repeats each model ``per_cell`` times onto the flat env axis;
        ``tiled=False`` keeps the ``(n_tradeoffs, group_size)`` group axes, so repetitions
        of a cell share one model. ``workers`` threads the per-design compiles; see
        :func:`build_batched_model`.
        """
        flat = self.designs.reshape(-1, self.designs.shape[-1])
        stacked = build_batched_model(env, flat, workers=workers)
        if not tiled:
            return jax.tree_util.tree_map(
                lambda x: x.reshape(self.n_tradeoffs, self.group_size, *x.shape[1:]),
                stacked,
            )
        return jax.tree_util.tree_map(
            lambda x: jax.numpy.repeat(x, self.per_cell, axis=0), stacked
        )

    def flatten(self) -> tuple[np.ndarray, np.ndarray]:
        """Tile designs/tradeoffs to the flat env axis ``((t * G) + g) * per_cell + c``.

        Returns ``(designs_full, tradeoffs_full)``, each with leading axis ``num_envs``.
        """
        designs_full = np.repeat(
            self.designs.reshape(-1, self.designs.shape[-1]), self.per_cell, axis=0
        )
        tradeoffs_full = np.repeat(
            self.tradeoffs, self.group_size * self.per_cell, axis=0
        )
        return designs_full, tradeoffs_full

    def group_view(self, x):
        """Reshape a flat env-axis array to ``(n_tradeoffs, group_size, per_cell, ...)``."""
        return x.reshape(
            self.n_tradeoffs, self.group_size, self.per_cell, *x.shape[1:]
        )

    def unflatten(self, x) -> np.ndarray:
        """Reshape a flat env-axis array to ``(group_size, n_tradeoffs, per_cell, ...)``."""
        return self.to_grid_axes(self.group_view(np.asarray(x)))

    def to_grid_axes(self, x) -> np.ndarray:
        """Map an array with :attr:`shape`'s leading axes to ``(design, tradeoff, rep, ...)``.

        Swaps the group-major axes so the result matches
        :class:`DesignTradeoffRolloutGrid`'s ``(design, tradeoff, rep)`` convention.
        """
        return np.swapaxes(np.asarray(x), 0, 1)


def _kept(n: int, sel) -> np.ndarray:
    """Indices kept along an axis of length ``n``.

    ``sel`` is any numpy index (slice, sequence, boolean mask, scalar) or ``None`` for
    the whole axis; tuples index as sequences, and scalars keep the axis.
    """
    if sel is None:
        sel = slice(None)
    elif isinstance(sel, tuple):
        sel = list(sel)
    return np.atleast_1d(np.arange(n)[sel])


@dataclasses.dataclass
class DesignTradeoffRolloutGrid:
    """Rollout results over a design x tradeoff grid, with the grid axes kept intact."""

    designs: np.ndarray            # (n_designs, design_dim), or (n_designs, n_tradeoffs,
                                   # design_dim) when each tradeoff has its own designs
    tradeoffs: np.ndarray          # (n_tradeoffs, num_objectives)
    rewards: np.ndarray            # (n_designs, n_tradeoffs, per_cell, num_objectives)
    objectives: list | None = None # per-objective names

    @classmethod
    def from_flat(
        cls, grid: DesignTradeoffSampleGrid, flat_rewards, objectives: list | None = None
    ) -> "DesignTradeoffRolloutGrid":
        """Build from per-env rewards laid out in ``grid``'s flat env ordering."""
        return cls(
            designs       = np.asarray(grid.designs),
            tradeoffs     = np.asarray(grid.tradeoffs),
            rewards       = grid.unflatten(flat_rewards),
            objectives    = objectives,
        )

    @property
    def mean_rewards(self) -> np.ndarray:
        """Rewards averaged over the per-cell repetition axis."""
        return self.rewards.mean(axis=2)

    def flatten(
        self, design_slice=None, tradeoff_slice=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drop the grid axes so rewards, designs, and tradeoffs index-correspond.
        """
        d_idx = _kept(self.rewards.shape[0], design_slice)
        t_idx = _kept(self.rewards.shape[1], tradeoff_slice)
        rewards = self.rewards[d_idx][:, t_idx]
        designs = self.designs[d_idx]
        # one design set shared by every tradeoff, or one per tradeoff
        designs = designs[:, None] if designs.ndim == 2 else designs[:, t_idx]
        grid = rewards.shape[:3]
        designs = np.broadcast_to(designs[:, :, None], grid + designs.shape[-1:])
        tradeoffs = np.broadcast_to(
            self.tradeoffs[t_idx][None, :, None], grid + self.tradeoffs.shape[-1:]
        )
        flat = lambda x: x.reshape(-1, x.shape[-1])
        return flat(rewards), flat(designs), flat(tradeoffs)

    def _arrays(self) -> dict[str, np.ndarray]:
        """The npz payload :meth:`save` writes."""
        arrays = dict(designs=self.designs, tradeoffs=self.tradeoffs, rewards=self.rewards)
        if self.objectives is not None:
            arrays["objectives"] = np.asarray(self.objectives, dtype=object)
        return arrays

    @classmethod
    def _fields(cls, npz) -> dict:
        """Constructor kwargs read back out of an npz written by :meth:`save`."""
        return dict(
            designs=npz["designs"],
            tradeoffs=npz["tradeoffs"],
            rewards=npz["rewards"],
            objectives=npz["objectives"].tolist() if "objectives" in npz else None,
        )

    def save(self, path: str | Path) -> None:
        np.savez(path, **self._arrays())

    @classmethod
    def load(cls, path: str | Path) -> "DesignTradeoffRolloutGrid":
        return cls(**cls._fields(np.load(path, allow_pickle=True)))

@dataclasses.dataclass
class DesignTradeoffDataset(DesignTradeoffRolloutGrid):
    """A :class:`DesignTradeoffRolloutGrid` plus the per-step trajectories behind it.

    Recorded ``data[key]`` carries the grid axes of ``rewards`` with a time axis in place
    of the objective axis: ``(n_designs, n_tradeoffs, per_cell, n_steps, ...)``. Additional
    per-cell values, such as an initial-state value prediction, omit the time axis.
    """

    data: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    config: dict | None = None

    _DATA_PREFIX = "data/"  # npz namespace keeping ``data`` apart from the grid arrays
    _CONFIG_KEY = "config"

    @classmethod
    def from_flat(
        cls,
        grid: DesignTradeoffSampleGrid,
        flat_rewards,
        objectives: list | None = None,
        flat_data: dict | None = None,
        config: dict | None = None,
    ) -> "DesignTradeoffDataset":
        """Build from per-env rewards and trajectories in ``grid``'s flat env ordering."""
        dataset = super().from_flat(grid, flat_rewards, objectives)
        dataset.data = {k: grid.unflatten(v) for k, v in (flat_data or {}).items()}
        dataset.config = config
        return dataset

    @property
    def keys(self) -> list[str]:
        """Names of the recorded per-step quantities."""
        return sorted(self.data)

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            **super()._arrays(),
            **{self._DATA_PREFIX + k: v for k, v in self.data.items()},
        }
        if self.config is not None:
            arrays[self._CONFIG_KEY] = np.asarray(self.config, dtype=object)
        return arrays

    @classmethod
    def _fields(cls, npz) -> dict:
        cut = len(cls._DATA_PREFIX)
        return dict(
            **super()._fields(npz),
            data={
                k[cut:]: npz[k] for k in npz.files if k.startswith(cls._DATA_PREFIX)
            },
            config=npz[cls._CONFIG_KEY].item() if cls._CONFIG_KEY in npz else None,
        )
