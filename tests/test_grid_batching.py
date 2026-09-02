"""What :class:`Grid`'s batching and sampling strategies actually guarantee.

The grid is one dense ``(M, K, ...)`` array pair: cell ``(m, k)`` records *which design met
which tradeoff*, and no field of the class says whether the layout is crossed (design ``m``
is the same in every column) or paired (design ``m, k`` was drawn for tradeoff ``k``).
Batching just slices one of those two axes, so the properties a caller relies on -- "every
cell in this batch carries the same design" above all -- are properties of the *layout* the
batched grid happened to have, not of the batching call. These tests pin down both halves:
what the split guarantees on its own, and what it only guarantees on a crossed grid.

Everything here runs on CPU with no MuJoCo compile: the ``from_*`` constructors only read
``design_limits`` and ``objectives`` off the env, so a stub stands in for the cheetah.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from codesign.utils.grid import (
    DesignTransition,
    Grid,
    _box_designs_cpu,
    _generator,
    sample_tradeoffs_cpu,
)

M, K, C, N_R, T = 6, 4, 2, 3, 5  # designs, tradeoffs, per-cell repeats, objectives, steps


class StubEnv:
    """The only two attributes ``Grid``'s samplers read off an env."""

    def __init__(self, design_dim=2, n_objectives=N_R):
        # (2, design_dim): lows then highs, matching CodesignBase.design_limits.
        self.design_limits = np.stack(
            [np.zeros(design_dim, np.float32), np.arange(1, design_dim + 1, dtype=np.float32)]
        )
        # Real envs hand back a list of one-element lists; keep that shape.
        self.objectives = [[f"obj{i}"] for i in range(n_objectives)]


@pytest.fixture
def env():
    return StubEnv()


# ---------------------------------------------------------------- fingerprinted grids
#
# Every array is filled with a value that decodes back to the cell it came from, so a
# batch can be checked for *alignment*: designs, tradeoffs, rewards, data and transitions
# must all have been permuted by the same index, or a cell would report two different
# origins.


def _cell_id(m, k, c=None, t=None):
    """A unique float per (m, k[, c][, t]), packed so each axis reads back out."""
    v = m * 1000.0 + k * 100.0
    if c is not None:
        v += c * 10.0
    if t is not None:
        v += t
    return v


def _fingerprint(shape, has_c=True, has_t=False):
    idx = np.indices(shape[: 2 + has_c + has_t]).astype(np.float32)
    m, k = idx[0], idx[1]
    v = m * 1000.0 + k * 100.0
    if has_c:
        v = v + idx[2] * 10.0
    if has_t:
        v = v + idx[2 + has_c]
    return np.broadcast_to(v.reshape(v.shape + (1,) * (len(shape) - v.ndim)), shape).copy()


def _transitions():
    """A DesignTransition whose leaves are (M, K, C, T, ...) fingerprints."""
    return DesignTransition(
        observation=_fingerprint((M, K, C, T, 7), has_t=True),
        action=_fingerprint((M, K, C, T, 3), has_t=True),
        reward=_fingerprint((M, K, C, T), has_t=True),
        design=_fingerprint((M, K, C, T, 2), has_t=True),
        tradeoff=_fingerprint((M, K, C, T, N_R), has_t=True),
        discount=_fingerprint((M, K, C, T), has_t=True),
        next_observation=_fingerprint((M, K, C, T, 7), has_t=True),
        extras={"raw_action": _fingerprint((M, K, C, T, 3), has_t=True)},
    )


@pytest.fixture
def crossed():
    """Crossed layout: design ``m`` is repeated down every tradeoff column."""
    designs = np.arange(M, dtype=np.float32)[:, None] + np.array([[0.0, 0.5]], np.float32)
    tradeoffs = np.eye(K, N_R, dtype=np.float32) + 0.01  # K distinct rows
    return dataclasses.replace(
        Grid.crossed(designs, tradeoffs, per_cell=C, objectives=[["a"], ["b"], ["c"]]),
        rewards=_fingerprint((M, K, C, N_R)),
        data={"qpos": _fingerprint((M, K, C, T, 9), has_t=True)},
        transitions=_transitions(),
    )


@pytest.fixture
def paired():
    """Paired layout: cell (m, k) holds its own design, the m-th member of group k."""
    designs = (
        np.arange(M, dtype=np.float32)[:, None, None] * 10.0
        + np.arange(K, dtype=np.float32)[None, :, None]
        + np.array([[[0.0, 0.5]]], np.float32)
    )
    tradeoffs = np.eye(K, N_R, dtype=np.float32) + 0.01
    return dataclasses.replace(
        Grid.paired(designs, tradeoffs, per_cell=C, objectives=[["a"], ["b"], ["c"]]),
        rewards=_fingerprint((M, K, C, N_R)),
        data={"qpos": _fingerprint((M, K, C, T, 9), has_t=True)},
        transitions=_transitions(),
    )


KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------- the headline guarantee


def test_batch_by_design_gives_one_design_per_batch_when_crossed(crossed):
    """``n_batches == M`` on a crossed grid: every cell of a batch is the same design.

    This is the property ``make_sgd_step(batching_strategy='design')`` leans on, and it
    holds because a crossed grid repeats design ``m`` down all ``K`` columns -- slicing
    axis 0 to width 1 then leaves a single distinct design row behind.
    """
    batches = crossed.batch_by_design(KEY, M)
    assert len(batches) == M
    for batch in batches:
        rows = np.asarray(batch.designs).reshape(-1, batch.design_dim)
        assert len(np.unique(rows, axis=0)) == 1, "batch mixes designs"
        # It is still the full tradeoff sweep, so the batch spans all K objectives mixes.
        assert batch.shape == (1, K, C, N_R)
        assert len(np.unique(np.asarray(batch.tradeoffs).reshape(-1, N_R), axis=0)) == K


def test_batch_by_design_does_not_give_one_design_per_batch_when_paired(paired):
    """The same call on a paired grid yields ``K`` distinct designs per batch.

    Batching cannot manufacture the guarantee: a paired grid draws a *different* design
    for each tradeoff column, so a width-1 slice of axis 0 is one group member per
    tradeoff, not one design. Callers wanting "one design per minibatch" need the grid to
    be crossed, which is why this is asserted rather than assumed.
    """
    for batch in paired.batch_by_design(KEY, M):
        rows = np.asarray(batch.designs).reshape(-1, batch.design_dim)
        assert len(np.unique(rows, axis=0)) == K


def test_batch_by_tradeoff_gives_one_tradeoff_per_batch(crossed):
    """The mirror image on axis 1: ``n_batches == K`` isolates a single tradeoff."""
    batches = crossed.batch_by_tradeoff(KEY, K)
    assert len(batches) == K
    for batch in batches:
        weights = np.asarray(batch.tradeoffs).reshape(-1, N_R)
        assert len(np.unique(weights, axis=0)) == 1, "batch mixes tradeoffs"
        assert batch.shape == (M, 1, C, N_R)
        # All M designs are still present -- the design axis was untouched.
        assert len(np.unique(np.asarray(batch.designs).reshape(-1, 2), axis=0)) == M


# ---------------------------------------------------------------- partition properties


@pytest.mark.parametrize("n_batches", [1, 2, 3, 6])
def test_batch_by_design_is_a_contiguous_partition(crossed, n_batches):
    """Unshuffled, batch ``i`` is exactly ``designs[i * w : (i + 1) * w]``."""
    width = M // n_batches
    batches = crossed.batch_by_design(KEY, n_batches)
    assert len(batches) == n_batches
    for i, batch in enumerate(batches):
        sl = slice(i * width, (i + 1) * width)
        assert batch.shape == (width, K, C, N_R)
        np.testing.assert_array_equal(np.asarray(batch.designs), crossed.designs[sl])
        np.testing.assert_array_equal(np.asarray(batch.rewards), crossed.rewards[sl])
        np.testing.assert_array_equal(
            np.asarray(batch.data["qpos"]), crossed.data["qpos"][sl]
        )


@pytest.mark.parametrize("n_batches", [1, 2, 4])
def test_batch_by_tradeoff_is_a_contiguous_partition(crossed, n_batches):
    width = K // n_batches
    for i, batch in enumerate(crossed.batch_by_tradeoff(KEY, n_batches)):
        sl = slice(i * width, (i + 1) * width)
        assert batch.shape == (M, width, C, N_R)
        np.testing.assert_array_equal(np.asarray(batch.designs), crossed.designs[:, sl])
        np.testing.assert_array_equal(np.asarray(batch.rewards), crossed.rewards[:, sl])


@pytest.mark.parametrize("shuffle", [False, True])
def test_batching_partitions_without_loss_or_duplication(paired, shuffle):
    """Every cell lands in exactly one batch, shuffled or not -- a permutation, not a draw."""
    for axis, n_batches, cat_axis in ((0, M, 0), (1, K, 1)):
        fn = paired.batch_by_design if axis == 0 else paired.batch_by_tradeoff
        rejoined = np.concatenate(
            [np.asarray(b.rewards) for b in fn(KEY, n_batches, shuffle)], axis=cat_axis
        )
        assert sorted(rejoined.ravel().tolist()) == sorted(paired.rewards.ravel().tolist())


def test_unshuffled_batching_reconstructs_the_grid_exactly(paired):
    """Concatenating the unshuffled batches back gives the original arrays, in order."""
    batches = paired.batch_by_design(KEY, 3)
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(b.designs) for b in batches], axis=0), paired.designs
    )
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(b.transitions.reward) for b in batches], axis=0),
        paired.transitions.reward,
    )


# ---------------------------------------------------------------- cell alignment


def _decode_m_k(value):
    """Recover ``(m, k)`` from a fingerprint value."""
    v = np.asarray(value)
    return (v // 1000).astype(int), ((v % 1000) // 100).astype(int)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_every_field_is_permuted_by_the_same_index(paired, axis, shuffle):
    """designs, tradeoffs, rewards, data and transitions must move together.

    Each array is fingerprinted with the ``(m, k)`` cell it came from. If ``_batch`` ever
    applied a different gather to one field -- a stale index, a wrong axis -- the batched
    fields would disagree about which cell a row is, which is exactly what is checked.
    """
    fn = paired.batch_by_design if axis == 0 else paired.batch_by_tradeoff
    n_batches = M if axis == 0 else K
    for batch in fn(KEY, n_batches, shuffle):
        # Rewards are (m, k, c, n_r); take one repeat and one objective to get (m, k).
        rm, rk = _decode_m_k(np.asarray(batch.rewards)[..., 0, 0])
        for leaf in jax.tree_util.tree_leaves(batch.transitions):
            lm, lk = _decode_m_k(np.asarray(leaf).reshape(*batch.rewards.shape[:3], -1)[..., 0, 0])
            np.testing.assert_array_equal(lm, rm)
            np.testing.assert_array_equal(lk, rk)
        dm, dk = _decode_m_k(np.asarray(batch.data["qpos"])[..., 0, 0, 0])
        np.testing.assert_array_equal(dm, rm)
        np.testing.assert_array_equal(dk, rk)
        # The design row itself encodes 10 * m + k, independently of the fingerprints.
        d = np.asarray(batch.designs)[..., 0]
        np.testing.assert_allclose(d, rm * 10.0 + rk, atol=1e-4)


# ---------------------------------------------------------------- shuffling


def test_shuffle_permutes_and_is_key_deterministic(crossed):
    """The same key gives the same split; a different key generally gives another."""
    a = crossed.batch_by_design(KEY, 3, shuffle=True)
    b = crossed.batch_by_design(KEY, 3, shuffle=True)
    c = crossed.batch_by_design(jax.random.PRNGKey(7), 3, shuffle=True)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x.designs), np.asarray(y.designs))
    orders = [
        np.concatenate([np.asarray(g.designs)[:, 0, 0] for g in batches])
        for batches in (a, c)
    ]
    assert not np.array_equal(*orders), "two keys produced the identical order"


def test_shuffle_actually_reorders(crossed):
    """Over many keys the shuffled order differs from the identity at least once.

    A single key could permute to the identity by chance (1/M! per draw), so the check is
    that *some* key moves things -- which fails loudly if ``shuffle`` is a no-op.
    """
    identity = crossed.designs[:, 0, 0]
    moved = False
    for seed in range(10):
        order = np.concatenate(
            [
                np.asarray(g.designs)[:, 0, 0]
                for g in crossed.batch_by_design(jax.random.PRNGKey(seed), 2, shuffle=True)
            ]
        )
        moved |= not np.array_equal(order, identity)
    assert moved


def test_unshuffled_ignores_the_key(crossed):
    """``shuffle=False`` is contiguous regardless of the key it was handed."""
    a = crossed.batch_by_design(jax.random.PRNGKey(1), 3)
    b = crossed.batch_by_design(jax.random.PRNGKey(2), 3)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x.designs), np.asarray(y.designs))


# ---------------------------------------------------------------- metadata and edges


def test_batches_keep_grid_metadata(crossed):
    for batch in crossed.batch_by_design(KEY, 2):
        assert batch.per_cell == C
        assert batch.objectives == crossed.objectives
        assert batch.n_r == N_R
        assert batch.design_dim == crossed.design_dim
        assert batch.num_envs == (M // 2) * K * C


def test_batching_a_sampled_grid_leaves_none_fields_none(env):
    """A grid that was sampled but never rolled out has no rewards to gather."""
    grid = Grid.from_uniform_sample(env, 0, n_tradeoffs=K, n_designs=M)
    for batch in grid.batch_by_design(KEY, 2):
        assert batch.rewards is None
        assert batch.transitions is None
        assert batch.data == {}
        assert batch.designs.shape == (M // 2, K, 2)


@pytest.mark.parametrize(
    "fn_name, n_batches", [("batch_by_design", 4), ("batch_by_tradeoff", 3)]
)
def test_indivisible_batch_count_raises(crossed, fn_name, n_batches):
    """M = 6 is not divisible by 4, K = 4 is not divisible by 3."""
    with pytest.raises(ValueError, match="do not divide"):
        getattr(crossed, fn_name)(KEY, n_batches)


def test_batching_preserves_values_and_dtype(crossed):
    """Gathering goes through ``jnp.take``, so batches come back as jax arrays."""
    batch = crossed.batch_by_design(KEY, 2)[0]
    assert isinstance(batch.designs, jax.Array)
    assert batch.designs.dtype == crossed.designs.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(batch.designs), crossed.designs[:3])


# ---------------------------------------------------------------- jit


@functools.partial(jax.jit, static_argnums=(2, 3))
def _jit_by_design(key, grid, n_batches, shuffle):
    return grid.batch_by_design(key, n_batches, shuffle)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _jit_by_tradeoff(key, grid, n_batches, shuffle):
    return grid.batch_by_tradeoff(key, n_batches, shuffle)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("jit_fn, eager_name, n_batches", [
    (_jit_by_design, "batch_by_design", 3),
    (_jit_by_tradeoff, "batch_by_tradeoff", 2),
])
def test_batching_is_jittable_and_matches_eager(paired, jit_fn, eager_name, n_batches, shuffle):
    """``n_batches`` and ``shuffle`` are static; the grid rides in as a pytree.

    ``Grid`` is registered as a pytree node whose children are the arrays and whose aux
    data is ``(per_cell, objectives)``, so a whole grid can cross a ``jit`` boundary and
    a *list* of grids can come back out. Both must trace, and the traced result must equal
    what the same call produces eagerly.
    """
    jitted = jit_fn(KEY, paired, n_batches, shuffle)
    eager = getattr(paired, eager_name)(KEY, n_batches, shuffle)
    assert len(jitted) == len(eager) == n_batches
    for j, e in zip(jitted, eager):
        for jl, el in zip(jax.tree_util.tree_leaves(j), jax.tree_util.tree_leaves(e)):
            np.testing.assert_array_equal(np.asarray(jl), np.asarray(el))


def test_jit_batching_traces_once_per_static_signature(paired):
    """Re-calling with the same shapes must hit the cache, not retrace.

    Batching sits inside the jitted SGD step, so a grid whose pytree structure or aux data
    shifted between calls would retrace the whole training step every time.
    """
    traces = []

    @functools.partial(jax.jit, static_argnums=(2,))
    def counted(key, grid, n_batches):
        traces.append(1)
        return grid.batch_by_design(key, n_batches)

    counted(KEY, paired, 2)
    counted(jax.random.PRNGKey(3), paired, 2)
    counted(KEY, dataclasses.replace(paired, rewards=paired.rewards + 1.0), 2)
    assert len(traces) == 1
    counted(KEY, paired, 3)  # a new static n_batches is a new signature
    assert len(traces) == 2


def test_jit_batching_composes_with_a_downstream_tree_map(paired):
    """The realistic use: batch, then stack the batches into a scan-ready leading axis."""

    @functools.partial(jax.jit, static_argnums=(2,))
    def stack_minibatches(key, grid, n_batches):
        batches = grid.batch_by_design(key, n_batches, True)
        return jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[b.transitions for b in batches]
        )

    out = stack_minibatches(KEY, paired, 3)
    assert out.reward.shape == (3, M // 3, K, C, T)
    # Nothing was dropped: the stacked minibatches still hold every original step.
    assert sorted(np.asarray(out.reward).ravel().tolist()) == sorted(
        paired.transitions.reward.ravel().tolist()
    )


def test_grid_survives_a_pytree_roundtrip(crossed):
    """flatten/unflatten is what jit does at the boundary; metadata must come back."""
    leaves, treedef = jax.tree_util.tree_flatten(crossed)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.per_cell == crossed.per_cell
    assert back.objectives == crossed.objectives
    np.testing.assert_array_equal(back.designs, crossed.designs)
    np.testing.assert_array_equal(back.data["qpos"], crossed.data["qpos"])


# ---------------------------------------------------------------- tradeoff sampling


def test_sample_tradeoffs_dense_lies_on_the_simplex():
    w = sample_tradeoffs_cpu(_generator(0), 32, N_R, sampling="dense")
    assert w.shape == (32, N_R) and w.dtype == np.float32
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert (w >= 0).all()
    # Dirichlet draws are interior, so no draw should be a corner.
    assert not np.any(np.isclose(w.max(axis=1), 1.0))


def test_sample_tradeoffs_sparse_heavytail_appends_the_corners():
    """``sparse-heavytail`` reserves the last ``n_r`` rows for the one-hot extremes.

    Those corners are the single-objective ends of the Pareto front; without them a
    Dirichlet-only sample almost surely never asks for "optimize run alone".
    """
    n = 8
    w = sample_tradeoffs_cpu(_generator(0), n, N_R, sampling="sparse-heavytail")
    assert w.shape == (n, N_R)
    np.testing.assert_array_equal(w[n - N_R :], np.eye(N_R, dtype=np.float32))
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert not np.any(np.isclose(w[: n - N_R].max(axis=1), 1.0))


def test_sample_tradeoffs_sparse_heavytail_when_corners_alone_overflow():
    """Asking for fewer tradeoffs than objectives truncates to the first corners."""
    w = sample_tradeoffs_cpu(_generator(0), 2, N_R, sampling="sparse-heavytail")
    np.testing.assert_array_equal(w, np.eye(N_R, dtype=np.float32)[:2])


def test_sample_tradeoffs_single_avg_is_the_uniform_weighting():
    w = sample_tradeoffs_cpu(_generator(0), 5, N_R, sampling="single-avg")
    np.testing.assert_allclose(w, np.full((5, N_R), 1.0 / N_R, np.float32))


def test_sample_tradeoffs_alpha_controls_concentration():
    """Dirichlet(alpha): alpha << 1 piles mass on the corners, alpha >> 1 on the centre.

    Measured as the mean largest weight -- 1.0 at a corner, 1/n_r at the barycentre.
    """
    peaky = sample_tradeoffs_cpu(_generator(0), 512, N_R, "dense", alpha=0.05).max(axis=1)
    flat = sample_tradeoffs_cpu(_generator(0), 512, N_R, "dense", alpha=50.0).max(axis=1)
    assert peaky.mean() > 0.9
    assert flat.mean() < 0.5


def test_sample_tradeoffs_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="not implemented"):
        sample_tradeoffs_cpu(_generator(0), 4, N_R, sampling="nonsense")


def test_sample_tradeoffs_is_seed_deterministic():
    a = sample_tradeoffs_cpu(_generator(3), 6, N_R)
    b = sample_tradeoffs_cpu(_generator(3), 6, N_R)
    np.testing.assert_array_equal(a, b)


def test_generator_accepts_int_array_or_live_stream():
    """``_generator`` normalizes the three seed forms callers actually pass."""
    np.testing.assert_array_equal(
        _generator(5).random(3), _generator(jnp.asarray(5)).random(3)
    )
    stream = _generator(5)
    first, second = stream.random(3), stream.random(3)
    # A live Generator is passed through unaltered, so it keeps advancing across calls.
    assert not np.array_equal(first, second)
    assert _generator(stream) is stream


# ---------------------------------------------------------------- design sampling


def test_box_designs_stay_inside_the_limits(env):
    low, high = np.asarray(env.design_limits)
    d = _box_designs_cpu(0, env, 64)
    assert d.shape == (64, len(low)) and d.dtype == np.float32
    assert (d >= low).all() and (d <= high).all()


def test_box_designs_are_space_filling(env):
    """Sobol beats i.i.d. uniform on discrepancy: check each dim covers every quartile.

    A 64-point Sobol sequence puts exactly 16 points in each quartile of each axis, which
    an i.i.d. uniform sample would only do by luck.
    """
    d = _box_designs_cpu(0, env, 64)
    low, high = np.asarray(env.design_limits)
    unit = (d - low) / (high - low)
    for dim in range(unit.shape[1]):
        counts = np.histogram(unit[:, dim], bins=4, range=(0, 1))[0]
        np.testing.assert_array_equal(counts, [16, 16, 16, 16])


def test_box_designs_honour_an_explicit_limits_override(env):
    """``limits`` replaces the env box -- used to sample a sub-box during training."""
    limits = (np.array([0.4, 0.4], np.float32), np.array([0.6, 0.6], np.float32))
    d = _box_designs_cpu(0, env, 16, limits=limits)
    assert (d >= 0.4).all() and (d <= 0.6).all()
    # The env box is [0, 1] x [0, 2], so the override genuinely narrowed the sample.
    assert not np.array_equal(d, _box_designs_cpu(0, env, 16))


def test_box_designs_are_seed_deterministic(env):
    np.testing.assert_array_equal(
        _box_designs_cpu(1, env, 8), _box_designs_cpu(1, env, 8)
    )
    assert not np.array_equal(_box_designs_cpu(1, env, 8), _box_designs_cpu(2, env, 8))


# ---------------------------------------------------------------- grid constructors


def test_crossed_broadcasts_both_axes(env):
    designs = np.arange(M * 2, dtype=np.float32).reshape(M, 2)
    tradeoffs = sample_tradeoffs_cpu(_generator(0), K, N_R)
    g = Grid.crossed(designs, tradeoffs, per_cell=C)
    assert g.shape == (M, K, C, N_R)
    # Every column holds the same design; every row holds the same tradeoff.
    np.testing.assert_array_equal(g.designs, np.broadcast_to(designs[:, None], (M, K, 2)))
    np.testing.assert_array_equal(g.tradeoffs, np.broadcast_to(tradeoffs[None], (M, K, N_R)))
    np.testing.assert_array_equal(g.unique_tradeoffs, tradeoffs)


def test_paired_keeps_per_cell_designs_and_broadcasts_tradeoffs(env):
    designs = np.arange(M * K * 2, dtype=np.float32).reshape(M, K, 2)
    tradeoffs = sample_tradeoffs_cpu(_generator(0), K, N_R)
    g = Grid.paired(designs, tradeoffs, per_cell=C)
    np.testing.assert_array_equal(g.designs, designs)
    np.testing.assert_array_equal(g.tradeoffs, np.broadcast_to(tradeoffs[None], (M, K, N_R)))
    # Column k is tradeoff k for every group member m -- the pairing the name promises.
    for k in range(K):
        np.testing.assert_array_equal(g.tradeoffs[:, k], np.broadcast_to(tradeoffs[k], (M, N_R)))


def test_from_design_sample_is_the_single_objective_layout(env):
    """One trivial tradeoff of weight 1.0 on an already-scalar reward: (M, 1, C, 1)."""
    g = Grid.from_design_sample(env, 0, n_designs=M, per_cell=C)
    assert g.shape == (M, 1, C, 1)
    np.testing.assert_array_equal(g.tradeoffs, np.ones((M, 1, 1), np.float32))
    low, high = np.asarray(env.design_limits)
    assert (g.designs >= low).all() and (g.designs <= high).all()


def test_from_design_sample_forwards_its_limits(env):
    """The ``limits`` argument must reach ``_box_designs_cpu``, not be dropped."""
    limits = (np.array([0.4, 0.4], np.float32), np.array([0.6, 0.6], np.float32))
    g = Grid.from_design_sample(env, 0, n_designs=8, limits=limits)
    assert (g.designs >= 0.4).all() and (g.designs <= 0.6).all()


def test_from_uniform_sample_is_crossed(env):
    g = Grid.from_uniform_sample(env, 0, n_tradeoffs=K, n_designs=M, per_cell=C)
    assert g.shape == (M, K, C, N_R)
    assert g.objectives == env.objectives
    np.testing.assert_array_equal(g.designs, np.broadcast_to(g.designs[:, :1], (M, K, 2)))
    # Default sampling is sparse-heavytail, so the last n_r tradeoffs are the corners.
    np.testing.assert_array_equal(g.unique_tradeoffs[K - N_R :], np.eye(N_R, dtype=np.float32))


def test_from_uniform_sample_is_seed_deterministic(env):
    a = Grid.from_uniform_sample(env, 0, n_tradeoffs=K, n_designs=M)
    b = Grid.from_uniform_sample(env, 0, n_tradeoffs=K, n_designs=M)
    np.testing.assert_array_equal(a.designs, b.designs)
    np.testing.assert_array_equal(a.tradeoffs, b.tradeoffs)


def test_from_simplex_corners_uses_exactly_the_one_hot_tradeoffs(env):
    g = Grid.from_simplex_corners(env, 0, n_designs=M, per_cell=C)
    assert g.shape == (M, N_R, C, N_R)
    np.testing.assert_array_equal(g.unique_tradeoffs, np.eye(N_R, dtype=np.float32))


def test_from_2d_tradeoffs_sweeps_each_objective_pair(env):
    """One linear sweep per unordered pair: C(n_r, 2) blocks of ``n_per_pair`` weights."""
    n_per_pair = 5
    g = Grid.from_2d_tradeoffs(env, 0, n_tradeoffs_per_pair=n_per_pair, n_designs=M)
    n_pairs = N_R * (N_R - 1) // 2
    assert g.n_tradeoffs == n_pairs * n_per_pair
    w = g.unique_tradeoffs
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    # Each block puts all its mass on one pair, leaving the other objectives at zero.
    for b in range(n_pairs):
        block = w[b * n_per_pair : (b + 1) * n_per_pair]
        assert (block > 0).any(axis=0).sum() == 2, "a block touched more than 2 objectives"


def test_from_2d_tradeoffs_needs_two_objectives():
    with pytest.raises(ValueError, match="at least 2 objectives"):
        Grid.from_2d_tradeoffs(StubEnv(n_objectives=1), 0, 3, 2)


# ---------------------------------------------------------------- flat env axis


def test_flatten_orders_envs_design_major_then_tradeoff_then_repeat(crossed):
    """The flat index is ``((m * K) + k) * per_cell + c`` -- the rollout's env ordering.

    ``build_models`` and ``env_inputs`` gather along this axis, so a mismatch here would
    silently hand env ``i`` another cell's design.
    """
    designs, tradeoffs = crossed.flatten()
    assert designs.shape == (M * K * C, 2)
    assert tradeoffs.shape == (M * K * C, N_R)
    for m in range(M):
        for k in range(K):
            for c in range(C):
                i = (m * K + k) * C + c
                np.testing.assert_array_equal(designs[i], crossed.designs[m, k])
                np.testing.assert_array_equal(tradeoffs[i], crossed.tradeoffs[m, k])


def test_unflatten_inverts_flatten(paired):
    designs, _ = paired.flatten()
    np.testing.assert_array_equal(
        paired.unflatten(designs),
        np.broadcast_to(paired.designs[:, :, None], (M, K, C, 2)),
    )


def test_flatten_then_batch_agree_on_which_design_is_where(crossed):
    """A design-batched grid flattens to the same rows as the slice of the full flatten."""
    full, _ = crossed.flatten()
    batches = crossed.batch_by_design(KEY, 2)
    per_batch = (M // 2) * K * C
    for i, batch in enumerate(batches):
        sub, _ = dataclasses.replace(
            batch, designs=np.asarray(batch.designs), tradeoffs=np.asarray(batch.tradeoffs)
        ).flatten()
        np.testing.assert_array_equal(sub, full[i * per_batch : (i + 1) * per_batch])
