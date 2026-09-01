"""Shape, model-building, batching, and serialization checks on :class:`Grid`."""

import dataclasses

import jax
import numpy as np
import pytest

import codesign.utils.grid as grid_mod
from codesign.utils.grid import Grid

N_DESIGNS, N_TRADEOFFS, PER_CELL, N_OBJECTIVES = 3, 4, 2, 3


@pytest.fixture(scope="module")
def crossed(mo_env):
    """Every design met by every tradeoff -- the layout usup samples."""
    return Grid.from_uniform_sample(
        mo_env, 0, n_tradeoffs=N_TRADEOFFS, n_designs=N_DESIGNS, per_cell=PER_CELL
    )


@pytest.fixture(scope="module")
def paired(mo_env, crossed):
    """One design per (design, tradeoff) cell -- the layout dpup's predictor produces."""
    designs = np.random.default_rng(0).uniform(
        0.5, 2.0, (N_DESIGNS, N_TRADEOFFS, 1)
    ).astype(np.float32)
    return Grid.paired(
        designs, crossed.unique_tradeoffs, per_cell=PER_CELL, objectives=mo_env.objectives
    )


def test_crossed_shapes(crossed):
    assert crossed.shape == (N_DESIGNS, N_TRADEOFFS, PER_CELL, N_OBJECTIVES)
    assert crossed.num_envs == N_DESIGNS * N_TRADEOFFS * PER_CELL
    # A crossed grid holds each design constant along tradeoffs, and vice versa.
    assert np.all(crossed.designs == crossed.designs[:, :1])
    assert np.all(crossed.tradeoffs == crossed.tradeoffs[:1])


def test_single_objective_grid(so_env):
    """design_hypernetwork's grid: one trivial tradeoff, one objective."""
    grid = Grid.from_design_sample(so_env, 0, n_designs=N_DESIGNS, per_cell=4)
    assert grid.shape == (N_DESIGNS, 1, 4, 1)
    assert np.all(grid.tradeoffs == 1.0)


def test_paired_shapes(paired, crossed):
    assert paired.shape == crossed.shape
    assert np.allclose(paired.tradeoffs, crossed.tradeoffs)


def test_flatten_unflatten_roundtrip(crossed):
    designs, tradeoffs = crossed.flatten()
    assert designs.shape == (crossed.num_envs, crossed.design_dim)
    assert tradeoffs.shape == (crossed.num_envs, N_OBJECTIVES)
    # Every one of a cell's per_cell repeats gets that cell's design back.
    assert np.array_equal(
        crossed.unflatten(designs)[..., 0],
        np.broadcast_to(crossed.designs[..., 0, None], crossed.cell_shape),
    )


@pytest.mark.slow
def test_build_models_compiles_only_unique_designs(mo_env, crossed, paired, monkeypatch):
    """A compile per distinct design row: 3 for a crossed 3x4, 12 for a paired one."""
    compiled = []
    original = grid_mod.build_batched_model

    def counting(env, designs, **kwargs):
        compiled.append(len(designs))
        return original(env, designs, **kwargs)

    monkeypatch.setattr(grid_mod, "build_batched_model", counting)
    crossed_models = crossed.build_models(mo_env, tiled=False)
    paired.build_models(mo_env, tiled=False)
    tiled = crossed.build_models(mo_env, tiled=True)

    assert compiled == [N_DESIGNS, N_DESIGNS * N_TRADEOFFS, N_DESIGNS]
    assert crossed_models.body_mass.shape[:2] == (N_DESIGNS, N_TRADEOFFS)
    assert tiled.body_mass.shape[0] == crossed.num_envs
    # The gathered per-cell models are the per-design ones, repeated along tradeoffs.
    geom_size = np.asarray(crossed_models.geom_size)
    assert np.allclose(geom_size[:, 0], geom_size[:, -1])


@pytest.mark.slow
def test_env_inputs_normalizes_designs(mo_env, crossed):
    model, designs, tradeoffs = crossed.env_inputs(mo_env)
    assert designs.shape == (crossed.num_envs, crossed.design_dim)
    assert tradeoffs.shape == (crossed.num_envs, N_OBJECTIVES)
    # The hypernetwork is conditioned on [0, 1] designs; the grid holds physical units.
    assert 0.0 <= float(designs.min()) and float(designs.max()) <= 1.0
    assert not np.allclose(np.asarray(designs), crossed.flatten()[0])


@pytest.mark.slow
def test_treedef_pinning_keeps_one_structure(mo_env, crossed):
    """Resampling must not change the model pytree, or every jitted fn retraces."""
    model = crossed.build_models(mo_env, tiled=True)
    again = crossed.build_models(mo_env, tiled=True, like=model)
    assert jax.tree_util.tree_structure(again) == jax.tree_util.tree_structure(model)


@pytest.fixture
def filled(crossed):
    """A rolled-out grid, with a recorded per-step quantity alongside its rewards."""
    cells = crossed.cell_shape
    return dataclasses.replace(
        crossed,
        rewards=np.arange(np.prod(crossed.shape), dtype=np.float32).reshape(crossed.shape),
        data={"qpos": np.zeros(cells + (5, 9), np.float32)},
    )


def test_batch_by_design_is_contiguous_without_shuffle(filled):
    batches = filled.batch_by_design(jax.random.PRNGKey(0), N_DESIGNS)
    for batch in batches:
        assert batch.shape == (1, N_TRADEOFFS, PER_CELL, N_OBJECTIVES)
        assert batch.data["qpos"].shape == (1, N_TRADEOFFS, PER_CELL, 5, 9)
    assert np.array_equal(
        np.concatenate([b.rewards for b in batches], axis=0), filled.rewards
    )


def test_batch_by_design_with_shuffle_permutes(filled):
    """A shuffled split still partitions the grid, just not in order."""
    shuffled = np.concatenate(
        [np.asarray(b.rewards)
         for b in filled.batch_by_design(jax.random.PRNGKey(1), N_DESIGNS, shuffle=True)],
        axis=0,
    )
    assert sorted(shuffled.ravel().tolist()) == sorted(filled.rewards.ravel().tolist())


def test_batch_by_tradeoff(filled):
    for batch in filled.batch_by_tradeoff(jax.random.PRNGKey(0), 2, shuffle=True):
        assert batch.shape == (N_DESIGNS, N_TRADEOFFS // 2, PER_CELL, N_OBJECTIVES)
        assert batch.data["qpos"].shape == (N_DESIGNS, N_TRADEOFFS // 2, PER_CELL, 5, 9)


def test_save_load_roundtrip(filled, tmp_path):
    path = tmp_path / "grid.npz"
    filled.save(path)
    back = Grid.load(path)
    for field in ("designs", "tradeoffs", "rewards"):
        assert np.array_equal(getattr(filled, field), getattr(back, field)), field
    assert back.per_cell == filled.per_cell
    assert back.objectives == filled.objectives
    assert back.keys == ["qpos"]
    assert np.array_equal(back.data["qpos"], filled.data["qpos"])


def test_save_load_sampled_grid(mo_env, tmp_path):
    """``rewards is None`` marks a grid that was sampled but never rolled out."""
    path = tmp_path / "sampled.npz"
    Grid.from_simplex_corners(mo_env, 0, n_designs=2).save(path)
    assert Grid.load(path).rewards is None
