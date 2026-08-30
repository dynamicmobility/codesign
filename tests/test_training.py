"""The three training algos end to end, plus the schedule arithmetic they share."""

import math

import pytest

import codesign
from codesign.hyperdesigners.shared import Schedule

# Small enough that an epoch is a few seconds, large enough to exercise every branch.
TINY = dict(
    episode_length=8, num_envs=8, unroll_length=2, batch_size=4, num_minibatches=2,
    num_updates_per_batch=1, num_eval_envs=8,
)
TINY["num_timesteps"] = TINY["batch_size"] * TINY["unroll_length"] * TINY["num_minibatches"]

ALGOS = {
    "design_hypernetwork": (
        codesign.setup_design_hypernetwork,
        dict(num_designs=2),
        (2, 1, 4, 1),  # (designs, tradeoffs, per_cell, objectives)
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
    assert schedule.num_evals_after_init == 4
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
    assert four.num_training_steps_per_chunk == math.ceil(
        one.num_training_steps_per_chunk / 4
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
        TINY["num_timesteps"], 3, TINY["num_envs"], TINY["batch_size"],
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
