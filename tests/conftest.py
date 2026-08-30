"""Fixtures and the ``--runslow`` gate for the codesign suite.

Tests that compile MuJoCo models or take training steps are marked ``slow`` and skipped
unless ``--runslow`` is given, so the default run stays quick.
"""

import functools
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

CONFIGS = {
    "design_hypernetwork": REPO / "config/design_hypernetwork/cheetah1D.yaml",
    "mo_design_hypernetwork": REPO / "config/mo_design_hypernetwork/cheetah1D.yaml",
    "mo_design_predictor_hypernetwork":
        REPO / "config/mo_design_predictor_hypernetwork/cheetah1D.yaml",
    "design_mlp": REPO / "config/design_mlp/cheetah3D.yaml",
}


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", help="also run tests marked slow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@functools.cache
def load_case(algorithm: str):
    """``(config, env)`` for one checked-in config.

    Cached because building a cheetah costs a ``spec.compile``, which dominates the
    runtime of every test needing an env.
    """
    import minimal_mjx as mm
    import moplayground as mop

    import codesign

    config = mm.create_config_dict(mop.utils.read_config(str(CONFIGS[algorithm])))
    env, _ = codesign.envs.create.load_env(config)
    # load_env scalarizes for design_hypernetwork and ppo but not design_mlp, which
    # scripts/train.py wraps itself; mirror that so the env matches the algo.
    if algorithm == "design_mlp":
        env = codesign.CodesignMO2SO(
            env, config["env_config"]["reward"]["optimization"]["default_scalarization"]
        )
    return config, env


@pytest.fixture(scope="session")
def so_env():
    """The single-objective cheetah: an MO env behind a ``CodesignMO2SO`` wrapper."""
    return load_case("design_hypernetwork")[1]


@pytest.fixture(scope="session")
def mo_env():
    """The multi-objective cheetah."""
    return load_case("mo_design_hypernetwork")[1]


@pytest.fixture(scope="session")
def case():
    """``algorithm -> (config, env)`` for the checked-in configs."""
    return load_case
