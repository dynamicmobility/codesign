"""The hypernetwork bundles: critic width and the extras the PPO losses read."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from codesign.hyperdesigners import (
    make_design_hypernet_networks,
    make_design_inference_fn,
    make_lookup_hypernet_networks,
    make_mo_design_hypernet_networks,
    make_mo_design_predictor_hypernet_networks,
    sobol_design_table,
    warn_off_table,
)

OBS, ACT, DESIGN_DIM, NUM_OBJECTIVES, BATCH = 24, 6, 1, 3, 5
NUM_DESIGNS, LOW, HIGH = 4, (0.5,), (2.0,)


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(0)


def critic_shape(bundle, cond_dim, key):
    """Shape of the critic's output, with per-env params drawn from the hypernetwork."""
    hyper_params = bundle.hypernetwork.init(key)
    _, value_params = bundle.hypernetwork.apply(hyper_params, jnp.zeros((BATCH, cond_dim)))
    apply = jax.vmap(bundle.value_network.apply, in_axes=(None, 0, 0))
    return apply(None, value_params, jnp.zeros((BATCH, OBS))).shape


@pytest.fixture(scope="module")
def single_objective(key):
    return make_design_hypernet_networks(
        observation_size=OBS, action_size=ACT, design_dim=DESIGN_DIM, key=key
    )


@pytest.fixture(scope="module")
def mo(key):
    return make_mo_design_hypernet_networks(
        observation_size=OBS, action_size=ACT, design_dim=DESIGN_DIM,
        num_objectives=NUM_OBJECTIVES, key=key,
    )


@pytest.fixture(scope="module")
def mo_predictor(key):
    return make_mo_design_predictor_hypernet_networks(
        observation_size=OBS, action_size=ACT, design_dim=DESIGN_DIM,
        num_objectives=NUM_OBJECTIVES, key=key,
    )


def test_single_objective_critic_is_scalar(single_objective, key):
    assert critic_shape(single_objective, DESIGN_DIM, key) == (BATCH,)


def test_mo_critic_has_one_output_per_objective(mo, key):
    """``compute_mo_design_hypernet_loss`` vmaps ``compute_gae`` over the objective axis,
    so the critic's width has to match the reward's."""
    assert critic_shape(mo, DESIGN_DIM + NUM_OBJECTIVES, key) == (BATCH, NUM_OBJECTIVES)


def test_predictor_critic_matches_mo(mo, mo_predictor, key):
    cond_dim = DESIGN_DIM + NUM_OBJECTIVES
    assert critic_shape(mo_predictor, cond_dim, key) == critic_shape(mo, cond_dim, key)


def test_design_inference_returns_raw_action(single_objective, key):
    """The PPO loss reads ``policy_extras['raw_action']``, so sampling must emit it."""
    inference_fn = make_design_inference_fn(single_objective)
    policy = inference_fn(
        (None, single_objective.hypernetwork.init(key)), jnp.zeros((BATCH, DESIGN_DIM))
    )
    _, extras = policy(jnp.zeros((BATCH, OBS)), key)
    assert set(extras) == {"log_prob", "raw_action"}


def test_deterministic_design_inference_has_no_extras(single_objective, key):
    inference_fn = make_design_inference_fn(single_objective)
    policy = inference_fn(
        (None, single_objective.hypernetwork.init(key)),
        jnp.zeros((BATCH, DESIGN_DIM)),
        deterministic=True,
    )
    assert policy(jnp.zeros((BATCH, OBS)), key)[1] == {}


# ------------------------------------------------------- design_lookup_hypernetwork
@pytest.fixture(scope="module")
def lookup(key):
    return make_lookup_hypernet_networks(
        observation_size=OBS, action_size=ACT, design_dim=DESIGN_DIM, key=key,
        num_designs=NUM_DESIGNS, design_seed=0, design_low=LOW, design_high=HIGH,
    )


def lookup_grads(bundle, params, design):
    """``d/dparams`` of a dummy loss on both heads the hypernetwork emits for ``design``."""
    obs = jnp.ones((len(design), OBS))

    def loss(hyper_params):
        policy_params, value_params = bundle.hypernetwork.apply(hyper_params, design)
        logits = jax.vmap(bundle.policy_network.apply, in_axes=(None, 0, 0))(
            None, policy_params, obs
        )
        values = jax.vmap(bundle.value_network.apply, in_axes=(None, 0, 0))(
            None, value_params, obs
        )
        return jnp.sum(logits ** 2) + jnp.sum(values ** 2)

    return jax.grad(loss)(params)["params"]


def test_lookup_feeds_one_row_of_w_per_design(lookup, key):
    """Row j of W takes gradient from design j alone, and the shared bias takes none."""
    params = lookup.hypernetwork.init(key)
    table = sobol_design_table(0, NUM_DESIGNS, LOW, HIGH)

    for j in range(NUM_DESIGNS):
        grads = lookup_grads(lookup, params, jnp.tile(jnp.asarray(table[j]), (BATCH, 1)))
        for name in ("policy_W", "value_W"):
            per_row = np.abs(np.asarray(grads[name])).sum(axis=1)
            assert per_row[j] > 0
            assert np.all(per_row[np.arange(NUM_DESIGNS) != j] == 0)
        for name in ("policy_b", "value_b"):
            assert np.all(np.asarray(grads[name]) == 0)


def test_lookup_has_no_feature_mlp(lookup, key):
    """The one-hot map is not learned, so the parent's feature MLPs are never built."""
    params = lookup.hypernetwork.init(key)
    assert set(params["params"]) == {"policy_W", "policy_b", "value_W", "value_b"}
    assert params["params"]["policy_W"].shape[0] == NUM_DESIGNS


def test_lookup_policy_is_bias_plus_its_own_row(lookup, key):
    """``b + W[i]`` is design i's whole parameter vector, so freezing b leaves W free."""
    params = lookup.hypernetwork.init(key)["params"]
    table = sobol_design_table(0, NUM_DESIGNS, LOW, HIGH)
    flat = lookup.hypernetwork.apply({"params": params}, jnp.asarray(table[2]))
    expected = params["policy_W"][2] + params["policy_b"]
    got = jnp.concatenate([
        x.reshape(-1) for _, x in sorted(
            jax.tree_util.tree_flatten_with_path(flat[0]["params"])[0],
            key=lambda kv: jax.tree_util.keystr(kv[0]),
        )
    ])
    assert np.allclose(np.asarray(got), np.asarray(expected))


def test_lookup_table_is_fixed_by_its_seed(lookup):
    """Sobol scrambles on ``(dim, seed)`` alone, so a longer draw extends a shorter one."""
    table = sobol_design_table(0, NUM_DESIGNS, LOW, HIGH)
    assert np.allclose(sobol_design_table(0, 2 * NUM_DESIGNS, LOW, HIGH)[:NUM_DESIGNS], table)
    assert np.all((table >= 0) & (table <= 1))


def test_lookup_critic_is_scalar(lookup, key):
    assert critic_shape(lookup, DESIGN_DIM, key) == (BATCH,)


def test_off_table_design_warns(key):
    """An off-table design silently gets a neighbour's policy, so it has to be flagged."""
    table = sobol_design_table(0, NUM_DESIGNS, LOW, HIGH)
    warn_off_table(jnp.asarray(table[1]), table)  # an anchor itself: nothing to say

    with pytest.warns(UserWarning, match="nearest anchor"):
        warn_off_table(jnp.full((1,), 10.0), table)


def test_on_table_designs_are_silent(key):
    table = sobol_design_table(0, NUM_DESIGNS, LOW, HIGH)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_off_table(jnp.asarray(table), table)
