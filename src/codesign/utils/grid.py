"""Design x tradeoff grid structures shared by training and evaluation.
"""

import dataclasses
from pathlib import Path

import numpy as np
import jax
from mujoco import mjx

from codesign.utils.model import stack_models, sample_designs, put_design_model
from codesign.envs.codesign_base import CodesignBase

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

@dataclasses.dataclass
class DesignTradeoffSampleGrid:
    """A sampled design x tradeoff grid and its flat env-axis ordering."""

    designs: np.ndarray      # (n_designs, design_dim)
    tradeoffs: np.ndarray    # (n_tradeoffs, num_objectives)
    per_cell: int = 1        # rollout repetitions per (design, tradeoff) cell
    
    @classmethod
    def from_uniform_sample(
        cls, env: CodesignBase, seed: int, n_tradeoffs, n_designs, per_cell: int = 1
    ) -> "DesignTradeoffSampleGrid":
        tradeoffs = jax.random.dirichlet(
            jax.random.PRNGKey(seed),
            alpha=np.ones(len(env.objectives)),
            shape=(n_tradeoffs,),
        )
        limits = np.asarray(env.design_limits)
        designs = sample_designs(
            np.random.default_rng(seed),
            n_designs,
            low=limits[0],
            high=limits[1],
            dim=limits.shape[1],
        )
        return cls(
            designs=jax.numpy.asarray(designs),
            tradeoffs=tradeoffs,
            per_cell=per_cell,
        )


    @property
    def n_designs(self) -> int:
        return self.designs.shape[0]

    @property
    def n_tradeoffs(self) -> int:
        return self.tradeoffs.shape[0]

    @property
    def num_envs(self) -> int:
        return self.n_designs * self.n_tradeoffs * self.per_cell

    def build_models(self, env: CodesignBase, tiled: bool) -> mjx.Model:
        """Stack one model per design (``tiled=False``) or per flat env (``tiled=True``)."""
        stacked = stack_models(put_design_model(env, d) for d in self.designs)
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

    @property
    def n_tradeoffs(self) -> int:
        return self.tradeoffs.shape[0]

    @property
    def group_size(self) -> int:
        return self.designs.shape[1]

    @property
    def num_envs(self) -> int:
        return self.n_tradeoffs * self.group_size * self.per_cell

    def build_models(self, env: CodesignBase) -> mjx.Model:
        """One model per ``(tradeoff, design)`` pair, repeated ``per_cell`` times."""
        flat = self.designs.reshape(-1, self.designs.shape[-1])
        stacked = stack_models(put_design_model(env, d) for d in flat)
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
        """Reshape a flat env-axis array to ``(group_size, n_tradeoffs, per_cell, ...)``.

        Group-major axes swapped so the result matches
        :class:`DesignTradeoffRolloutGrid`'s ``(design, tradeoff, rep)`` convention.
        """
        return np.swapaxes(self.group_view(np.asarray(x)), 0, 1)


@dataclasses.dataclass
class DesignTradeoffRolloutGrid:
    """Rollout results over a design x tradeoff grid, with the grid axes kept intact."""

    designs: np.ndarray            # (n_designs, design_dim)
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

    def save(self, path: str | Path) -> None:
        arrays = dict(designs=self.designs, tradeoffs=self.tradeoffs, rewards=self.rewards)
        if self.objectives is not None:
            arrays["objectives"] = np.asarray(self.objectives, dtype=object)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "DesignTradeoffRolloutGrid":
        data = np.load(path, allow_pickle=True)
        objectives = data["objectives"].tolist() if "objectives" in data else None
        return cls(
            designs=data["designs"],
            tradeoffs=data["tradeoffs"],
            rewards=data["rewards"],
            objectives=objectives,
        )
