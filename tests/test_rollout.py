"""The grid rollout must hand every cell its own design, tradeoff, and model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from codesign.eval.parallel_eval import _named_records, build_grid_rollout_fn
from codesign.utils.grid import Grid

N_DESIGNS, N_TRADEOFFS, PER_CELL, N_STEPS = 2, 3, 2, 3

pytestmark = pytest.mark.slow


@pytest.fixture(params=["crossed", "paired"])
def grid(request, mo_env):
    if request.param == "crossed":
        return Grid.from_uniform_sample(
            mo_env, 0, n_tradeoffs=N_TRADEOFFS, n_designs=N_DESIGNS, per_cell=PER_CELL
        )
    designs = np.random.default_rng(0).uniform(
        0.5, 2.0, (N_DESIGNS, N_TRADEOFFS, 1)
    ).astype(np.float32)
    return Grid.paired(
        designs, np.eye(N_TRADEOFFS, dtype=np.float32), per_cell=PER_CELL,
        objectives=mo_env.objectives,
    )


def test_each_cell_gets_its_own_design_and_tradeoff(mo_env, grid):
    """The policy writes its (design, tradeoff) into the action, so recording the action
    shows which pair each cell actually ran."""

    def make_policy(params, designs, tradeoffs, deterministic):
        tag = designs[0] + 10.0 * tradeoffs[0]
        return lambda obs, key: (jnp.full(mo_env.action_size, tag), {})

    rollout = build_grid_rollout_fn(
        mo_env, N_STEPS, make_policy, record_fn=lambda prev, act, nst: {"action": act}
    )
    keys = jax.random.split(jax.random.PRNGKey(0), grid.num_envs)
    (_, _, rewards), records = rollout(
        keys.reshape(*grid.cell_shape, -1),
        jnp.asarray(grid.designs),
        jnp.asarray(grid.tradeoffs),
        grid.build_models(mo_env, tiled=False),
        None,
    )

    actions = _named_records(records)["action"]
    assert actions.shape == (*grid.cell_shape, N_STEPS, mo_env.action_size)
    expected = grid.designs[..., 0] + 10.0 * grid.tradeoffs[..., 0]
    assert np.allclose(actions[:, :, 0, 0, 0], expected, atol=1e-5)
    assert np.asarray(rewards).shape == grid.shape


def test_tiled_model_order_matches_flatten(mo_env):
    """Training rolls out on the flat env axis, so the tiled models must be in the order
    ``flatten()`` puts the designs in."""
    grid = Grid.from_uniform_sample(
        mo_env, 0, n_tradeoffs=N_TRADEOFFS, n_designs=N_DESIGNS, per_cell=PER_CELL
    )
    cells = np.asarray(grid.build_models(mo_env, tiled=False).geom_size)
    tiled = np.asarray(grid.build_models(mo_env, tiled=True).geom_size)
    assert np.allclose(
        tiled, np.repeat(cells.reshape(-1, *cells.shape[2:]), grid.per_cell, axis=0)
    )
    designs, _ = grid.flatten()
    assert np.allclose(
        designs, np.repeat(grid.designs.reshape(-1, grid.design_dim), grid.per_cell, axis=0)
    )
