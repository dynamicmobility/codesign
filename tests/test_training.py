"""The training algos end to end, plus the schedule arithmetic they share."""

import functools
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics

import codesign
from codesign.hyperdesigners.shared import Schedule
from codesign.utils.grid import DesignTransition, Grid
from codesign.utils.model import observation_spec

# Small enough that an epoch is a few seconds, large enough to exercise every branch.
TINY = dict(
    episode_length=8, num_parallel_envs=8, unroll_length=2, batch_size=4, num_minibatches=2,
    num_updates_per_batch=1, num_eval_envs=8,
)
TINY["num_timesteps"] = TINY["batch_size"] * TINY["unroll_length"] * TINY["num_minibatches"]

ALGOS = {
    "design_mlp": (
        codesign.setup_design_mlp,
        dict(num_designs=2),
        (2, 1, 4, 1),
    ),
    "design_hypernetwork": (
        codesign.setup_design_hypernetwork,
        dict(num_designs=2),
        (2, 1, 4, 1),  # (designs, tradeoffs, per_cell, objectives)
    ),
    "design_lookup_hypernetwork": (
        codesign.setup_design_lookup_hypernetwork,
        dict(num_designs=2),
        (2, 1, 4, 1),
    ),
    "mo_design_hypernetwork": (
        codesign.setup_mo_design_hypernetwork,
        dict(num_designs=2, num_tradeoffs=2, num_eval_designs=2, num_eval_tradeoffs=2),
        (2, 2, 2, 3),
    ),
    "mo_design_predictor_hypernetwork": (
        codesign.setup_mo_design_predictor_hypernetwork,
        # The config's warmup spans more epochs than these runs have, so pin it; the
        # warmup branch gets its own test below.
        dict(num_designs=2, num_tradeoffs=2, num_eval_designs=2, num_eval_tradeoffs=2,
             num_warmup_iters=0),
        (2, 2, 2, 3),
    ),
}


def train(case, algorithm, num_evals=3, **overrides):
    """Run ``algorithm`` for a couple of epochs; returns ``(metrics, progress_calls)``."""
    setup, extra, _ = ALGOS[algorithm]
    config, env = case(algorithm)
    train_fn, _ = setup(config)

    calls = []
    _, _, metrics = train_fn(
        environment=env,
        progress_fn=lambda step, m: calls.append((step, m)),
        run_evals=True,
        num_evals=num_evals,
        **{**TINY, **extra, **overrides},
    )
    return metrics, calls


# ----------------------------------------------------------------- schedule arithmetic

def test_schedule_splits_the_step_budget():
    schedule = Schedule.make(
        num_timesteps=100_000, num_evals=5, num_envs=128, batch_size=64,
        num_minibatches=2, unroll_length=20,
    )
    assert schedule.env_step_per_training_step == 64 * 20 * 2
    assert schedule.num_epochs == 4
    # ceil(100000 / (4 epochs * 2560 steps)) = 10 training steps per epoch.
    assert schedule.num_training_steps_per_epoch == 10
    assert schedule.env_step_per_epoch == 10 * 2560


def test_schedule_divides_an_epoch_among_resamples():
    """Resampling splits the epoch's budget rather than multiplying the work.

    Each chunk's step count is rounded up, so an epoch overshoots by less than one step
    per resample.
    """
    one = Schedule.make(100_000, 5, 128, 64, 2, 20, resamples_per_epoch=1)
    four = Schedule.make(100_000, 5, 128, 64, 2, 20, resamples_per_epoch=4)
    assert four.num_training_steps_per_resample == math.ceil(
        one.num_training_steps_per_resample / 4
    )
    assert one.num_training_steps_per_epoch <= four.num_training_steps_per_epoch
    assert four.num_training_steps_per_epoch < one.num_training_steps_per_epoch + 4


def test_schedule_rejects_indivisible_batch():
    with pytest.raises(AssertionError):
        Schedule.make(1000, 5, num_envs=128, batch_size=64, num_minibatches=1,
                      unroll_length=20)


# ------------------------------------------------------------------------- end to end

@pytest.mark.slow
@pytest.mark.parametrize("algorithm", list(ALGOS))
def test_algorithm_trains_and_evaluates(case, algorithm):
    expected_shape = ALGOS[algorithm][2]
    metrics, calls = train(case, algorithm)

    assert [step for step, _ in calls] == [0, calls[1][0], 2 * calls[1][0]]
    assert "eval/episode_reward" in metrics

    grid = metrics["eval_grid"]
    assert grid.shape == expected_shape
    assert grid.rewards is not None and grid.rewards.shape == grid.shape
    assert grid.designs.shape[:2] == grid.rewards.shape[:2]


@pytest.mark.slow
@pytest.mark.parametrize("resamples", [1, 2])
def test_design_hypernetwork_resamples_per_epoch(case, resamples):
    """Resampling mid-epoch redraws the designs; the step counter tracks the schedule."""
    _, calls = train(case, "design_hypernetwork", resamples_per_epoch=resamples)
    steps = [step for step, _ in calls]
    schedule = Schedule.make(
        TINY["num_timesteps"], 3, TINY["num_parallel_envs"], TINY["batch_size"],
        TINY["num_minibatches"], TINY["unroll_length"], resamples,
    )
    assert steps == [0, schedule.env_step_per_epoch, 2 * schedule.env_step_per_epoch]


@pytest.mark.slow
def test_predictor_reports_grpo_metrics_only_after_warmup(case):
    """Warmup designs come from a Sobol sample, not from ``f``, so there is no policy
    ratio to update the predictor on and no GRPO loss to report."""
    _, calls = train(
        case, "mo_design_predictor_hypernetwork", num_evals=4, num_warmup_iters=1
    )
    warmup, grpo = calls[1][1], calls[2][1]

    for metrics in (warmup, grpo):
        assert "training/warmup" in metrics
        assert "training/design_value" in metrics
        assert any(k.startswith("eval/design_mode_obj") for k in metrics)

    assert warmup["training/warmup"] == 1.0 and grpo["training/warmup"] == 0.0
    assert "training/design_total_loss" not in warmup
    assert "training/design_total_loss" in grpo
    assert "training/design_design_entropy" in grpo


# ------------------------------------------------------- design_lookup_hypernetwork
LOOKUP = dict(obs=12, act=3, designs=4, per_cell=3, steps=4, low=(0.5,), high=(2.0,))


def lookup_bundle():
    return codesign.make_lookup_hypernet_networks(
        observation_size=LOOKUP["obs"], action_size=LOOKUP["act"], design_dim=1,
        key=jax.random.PRNGKey(0), num_designs=LOOKUP["designs"],
        design_seed=0, design_low=LOOKUP["low"], design_high=LOOKUP["high"],
    )


def lookup_grid(reward_scale):
    """A grid of fake rollouts, one reward scale per design."""
    m, c, t = LOOKUP["designs"], LOOKUP["per_cell"], LOOKUP["steps"]
    table = codesign.sobol_design_table(0, m, LOOKUP["low"], LOOKUP["high"])
    cell = (m, 1, c, t)
    fill = lambda seed, *tail: jax.random.normal(jax.random.PRNGKey(seed), cell + tail)
    scale = jnp.asarray(reward_scale).reshape(m, 1, 1, 1)
    transitions = DesignTransition(
        observation=fill(1, LOOKUP["obs"]),
        action=fill(2, LOOKUP["act"]),
        reward=fill(3) * scale,
        design=jnp.broadcast_to(jnp.asarray(table).reshape(m, 1, 1, 1, 1), cell + (1,)),
        tradeoff=jnp.ones(cell + (1,)),
        discount=jnp.ones(cell),
        next_observation=fill(4, LOOKUP["obs"]),
        extras={
            "state_extras": {"truncation": jnp.zeros(cell)},
            "policy_extras": {
                "log_prob": fill(5), "raw_action": fill(6, LOOKUP["act"])
            },
        },
    )
    return Grid(
        designs=np.asarray(table).reshape(m, 1, 1),
        tradeoffs=np.ones((m, 1, 1), np.float32),
        per_cell=c,
        transitions=transitions,
    )


def lookup_step(bundle, reward_scale):
    """One SGD step over the whole grid; returns the updated hypernetwork params."""
    optimizer = codesign.hyperdesigners.shared.make_optimizer(1e-2, max_grad_norm=1.0)
    loss_fn = functools.partial(
        codesign.compute_design_hypernet_loss, design_networks=bundle
    )
    sgd_step = codesign.hyperdesigners.shared.make_sgd_step(
        loss_fn, optimizer, LOOKUP["designs"], "design"
    )
    params = codesign.DesignHypernetParams(
        hypernetwork=bundle.hypernetwork.init(jax.random.PRNGKey(0))
    )
    carry = (optimizer.init(params), params, jax.random.PRNGKey(7))
    (_, params, _), _ = sgd_step(
        carry, None, data=lookup_grid(reward_scale), normalizer_params=None
    )
    return params.hypernetwork["params"]


def test_lookup_sgd_step_touches_only_the_perturbed_design():
    """Doubling one design's rewards moves that design's row of W and nothing else.

    Exact under Adam and global-norm clipping: the shared key fixes which minibatch each
    design lands in, so every other row sees bit-identical gradients, and the clip factor
    at the perturbed row's step is set by that row alone.
    """
    m, perturbed = LOOKUP["designs"], 2
    bundle = lookup_bundle()
    base = lookup_step(bundle, [1.0] * m)
    bumped = lookup_step(bundle, [2.0 if j == perturbed else 1.0 for j in range(m)])

    for name in ("policy_W", "value_W"):
        moved = np.abs(np.asarray(base[name]) - np.asarray(bumped[name])).sum(axis=1)
        assert moved[perturbed] > 0
        assert np.all(moved[np.arange(m) != perturbed] == 0)
    for name in ("policy_b", "value_b"):
        assert np.array_equal(np.asarray(base[name]), np.asarray(bumped[name]))


def test_lookup_checkpoint_config_carries_the_design_table(case):
    """brax JSON-encodes the factory's kwargs, and a function there reloads as an unusable
    string, so the table and the initializer must both travel as plain data."""
    config, env = case("design_lookup_hypernetwork")
    _, network_factory = codesign.setup_design_lookup_hypernetwork(config)
    written = json.loads(
        brax_checkpoint.network_config(
            observation_size=env.observation_size, action_size=env.action_size,
            normalize_observations=True, network_factory=network_factory,
        ).to_json_best_effort()
    )["network_factory_kwargs"]

    codesign_cfg = config["env_config"]["codesign"]
    assert written["num_designs"] == config["learning_params"]["design_sampling"]["num_designs"]
    assert written["design_seed"] == config["learning_params"]["ppo_params"]["seed"]
    assert written["design_low"] == list(codesign_cfg["low"])
    assert written["design_high"] == list(codesign_cfg["high"])
    assert written["weight_initializer"] == config["learning_params"]["network_params"]["weight_initializer"]


def warmup_checkpoint(tmp_path, bundle, params):
    """A checkpoint dir shaped like a finished warm-up run, plus its ``config.yaml``."""
    normalizer = running_statistics.update(
        running_statistics.init_state(observation_spec(LOOKUP["obs"])),
        jax.random.normal(jax.random.PRNGKey(9), (32, LOOKUP["obs"])),
    )
    brax_checkpoint.save(
        path=tmp_path, step=42, params=(normalizer, params),
        config=brax_checkpoint.network_config(
            observation_size=LOOKUP["obs"], action_size=LOOKUP["act"],
            normalize_observations=True,
            network_factory=functools.partial(codesign.make_lookup_hypernet_networks),
        ),
        config_fname="ppo_network_config.json",
    )
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "env_config": {"codesign": {
            "low": list(LOOKUP["low"]), "high": list(LOOKUP["high"])
        }},
        "learning_params": {
            "ppo_params": {"seed": 0},
            "design_sampling": {"num_designs": LOOKUP["designs"]},
        },
    }))
    return tmp_path / f"{42:012d}", normalizer


def test_load_network_transfers_the_warm_up_policies(tmp_path):
    """A transferred run starts at the warm-up's policies, not a random blend of them.

    Copying ``W`` and ``b`` alone would not achieve that: the full run's feature MLP is
    random, so ``mlp(d) @ W`` is an arbitrary combination of the ``M`` experts. The MLPs are
    regressed onto the one-hot map first, and this measures what that buys.
    """
    m = LOOKUP["designs"]
    warm_bundle = lookup_bundle()
    warm_params = warm_bundle.hypernetwork.init(jax.random.PRNGKey(3))
    path, normalizer = warmup_checkpoint(tmp_path, warm_bundle, warm_params)

    full = codesign.make_design_hypernet_networks(
        observation_size=LOOKUP["obs"], action_size=LOOKUP["act"], design_dim=1,
        key=jax.random.PRNGKey(0), hypersize=(32, 32), num_features=m,
    )
    optimizer = codesign.hyperdesigners.shared.make_optimizer(1e-3)
    fresh = codesign.hyperdesigners.shared.init_training_state(
        codesign.DesignHypernetParams(hypernetwork=full.hypernetwork.init(jax.random.PRNGKey(1))),
        optimizer, LOOKUP["obs"],
    )
    merged = codesign.load_warmup_params(fresh, str(path), optimizer, quiet=True)

    # W and b come across untouched.
    got, want = merged.params.hypernetwork["params"], warm_params["params"]
    for name in ("policy_W", "policy_b", "value_W", "value_b"):
        assert np.array_equal(np.asarray(got[name]), np.asarray(want[name]))
    assert np.array_equal(
        np.asarray(merged.normalizer_params.mean), np.asarray(normalizer.mean)
    )
    assert merged.normalizer_params.count == normalizer.count

    # The policy at each anchor design matches the warm-up's, where a fresh feature MLP
    # gives an arbitrary blend of the experts.
    table = codesign.sobol_design_table(0, m, LOOKUP["low"], LOOKUP["high"])
    flatten = lambda tree: np.concatenate(
        [np.asarray(x).reshape(-1) for x in jax.tree_util.tree_leaves(tree)]
    )
    for design in map(jnp.asarray, table):
        target = flatten(warm_bundle.hypernetwork.apply(warm_params, design))
        transferred = flatten(full.hypernetwork.apply(merged.params.hypernetwork, design))
        untransferred = flatten(full.hypernetwork.apply(fresh.params.hypernetwork, design))
        scale = np.linalg.norm(target)
        assert np.linalg.norm(transferred - target) / scale < 0.01
        assert np.linalg.norm(untransferred - target) / scale > 0.5


def rolled_out_grid(design_dim, num_designs=4, per_cell=5):
    """An eval grid with per-design rewards, as ``shared.eval_metrics`` leaves it."""
    designs = np.linspace(0.5, 2.0, num_designs * design_dim, dtype=np.float32)
    rewards = np.arange(num_designs * per_cell, dtype=np.float32)
    return Grid(
        designs=designs.reshape(num_designs, 1, design_dim),
        tradeoffs=np.ones((num_designs, 1, 1), np.float32),
        per_cell=per_cell,
        rewards=rewards.reshape(num_designs, 1, per_cell, 1),
    )


def test_paired_eval_keys_repeat_across_designs():
    """Trial c gets the same key under every design, laid out design-major."""
    m, c = 4, 3
    keys = codesign.hyperdesigners.paired_eval_keys(jax.random.PRNGKey(0), m, c)
    assert keys.shape[0] == m * c
    blocks = [np.asarray(keys[i * c:(i + 1) * c]) for i in range(m)]
    for block in blocks[1:]:
        assert np.array_equal(block, blocks[0])
    assert len({tuple(np.asarray(k).reshape(-1)) for k in blocks[0]}) == c


def test_per_design_metrics_report_every_design():
    grid = rolled_out_grid(design_dim=1)
    metrics = codesign.hyperdesigners.per_design_metrics(grid)
    per_design = [metrics[f"eval/design{i}/episode_reward"] for i in range(grid.n_designs)]
    assert len(per_design) == grid.n_designs
    assert per_design == pytest.approx(
        list(np.asarray(grid.rewards)[:, 0, :, 0].mean(axis=1))
    )
    assert metrics["eval/worst_design_reward"] == pytest.approx(min(per_design))
    assert metrics["eval/best_design_reward"] == pytest.approx(max(per_design))


@pytest.mark.parametrize("design_dim", [1, 3])
def test_design_rewards_progress_writes_a_figure(tmp_path, design_dim):
    """The plot has to cope with a design too wide to put on an axis, so it falls back to
    the design index."""
    import matplotlib
    matplotlib.use("Agg")

    training_data = codesign.MODesignTrainingPlottingInfo(start_time=0.0)
    for step in (0, 100):
        codesign.plot_design_rewards_progress(
            num_steps=step, metrics={"eval_grid": rolled_out_grid(design_dim)},
            training_data=training_data, times=[], save_dir=tmp_path, run=None,
        )
    assert (tmp_path / "progress.svg").stat().st_size > 0
    assert (tmp_path / "design_rewards_progress.csv").exists()
    assert len(training_data.grids) == 2
