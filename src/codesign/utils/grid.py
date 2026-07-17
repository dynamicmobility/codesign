"""Design x tradeoff grid structures shared by training and evaluation.
"""

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
from mujoco import mjx

from codesign.utils.model import stack_models


@dataclasses.dataclass
class DesignTradeoffSampleGrid:
    """A sampled design x tradeoff grid and its flat env-axis ordering."""

    designs: np.ndarray      # (n_designs, design_dim)
    tradeoffs: np.ndarray    # (n_tradeoffs, num_objectives)
    per_cell: int = 1        # rollout repetitions per (design, tradeoff) cell

    @property
    def n_designs(self) -> int:
        return self.designs.shape[0]

    @property
    def n_tradeoffs(self) -> int:
        return self.tradeoffs.shape[0]

    @property
    def num_envs(self) -> int:
        return self.n_designs * self.n_tradeoffs * self.per_cell

    def build_models(
        self, generate_model_fn: Callable[[np.ndarray], mjx.Model], tiled: bool
    ) -> mjx.Model:
        """Stack one model per design (``tiled=False``) or per flat env (``tiled=True``)."""
        models = [generate_model_fn(np.asarray(d)) for d in self.designs]
        if tiled:
            reps = self.n_tradeoffs * self.per_cell
            models = [m for m in models for _ in range(reps)]
        return stack_models(models)

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
