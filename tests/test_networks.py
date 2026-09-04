"""The hypernetwork bundles: critic width and the extras the PPO losses read."""

import jax
import jax.numpy as jnp
import pytest

from codesign.hyperdesigners import (
    make_design_hypernet_networks,
    make_design_inference_fn,
    make_mo_design_hypernet_networks,
    make_mo_design_predictor_hypernet_networks,
)

OBS, ACT, DESIGN_DIM, NUM_OBJECTIVES, BATCH = 24, 6, 1, 3, 5


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
