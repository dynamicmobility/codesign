"""Continuing a run: what the second job inherits from the first."""

import jax
import numpy as np
import pandas as pd
import pytest
from brax.training.agents.ppo import checkpoint as brax_checkpoint

import minimal_mjx as mm
import codesign
from codesign.hyperdesigners import shared
from codesign.hyperdesigners.shared import Schedule

# Two epochs of the smallest run that still exercises the loop.
TINY = dict(
    episode_length=8, num_parallel_envs=8, unroll_length=2, batch_size=4,
    num_minibatches=2, num_updates_per_batch=1, num_eval_envs=8, num_evals=3,
    num_timesteps=4 * 2 * 2 * 2,
)


# ------------------------------------------------------------ finding the resume point

def test_find_resume_reads_the_step_and_epoch_off_the_checkpoints(tmp_path):
    for step in (0, 16, 32):
        (tmp_path / f"{step:012d}").mkdir()
    (tmp_path / "progress.csv").touch()

    resumed = mm.find_resume(tmp_path)
    assert resumed.step == 32
    # One checkpoint per epoch, after the one written before the first epoch.
    assert resumed.epoch == 2
    assert resumed.path.name == "000000000032"


def test_find_resume_is_none_before_the_first_checkpoint(tmp_path):
    assert mm.find_resume(tmp_path) is None
    assert mm.find_resume(tmp_path / "never-created") is None


def test_apply_resume_asks_only_for_the_steps_still_owed(tmp_path):
    (tmp_path / f"{32:012d}").mkdir()
    config = mm.create_config_dict(
        {"learning_params": {"ppo_params": {"num_timesteps": 100}}}
    )
    out = mm.apply_resume(config, mm.find_resume(tmp_path))

    assert out.learning_params.ppo_params.num_timesteps == 68
    assert out.learning_params.resume.step == 32
    # The caller's config is untouched, so the run directory keeps the full budget.
    assert config.learning_params.ppo_params.num_timesteps == 100


def test_apply_resume_refuses_a_budget_already_spent(tmp_path):
    (tmp_path / f"{100:012d}").mkdir()
    config = mm.create_config_dict(
        {"learning_params": {"ppo_params": {"num_timesteps": 100}}}
    )
    with pytest.raises(ValueError, match="raise num_timesteps"):
        mm.apply_resume(config, mm.find_resume(tmp_path))


# ------------------------------------------------------------------ replaying the epochs

class _Stop(Exception):
    """Raised from the initial eval, which runs just after the replay."""


def test_run_training_replays_the_sampler_through_the_epochs_already_done():
    """The design schedule keeps counting, so a phase or a noise level carries on."""
    seen = []

    def sample(it, extra_state, key):
        seen.append(it)
        return None, None

    def evaluate(*args):
        raise _Stop

    schedule = Schedule.make(
        num_timesteps=64, num_evals=3, num_envs=8, batch_size=4,
        num_minibatches=2, unroll_length=2, resamples_per_epoch=2,
    )
    with pytest.raises(_Stop):
        shared.run_training(
            shared.Algorithm(
                sample=sample, chunk=None, evaluate=evaluate, params_of=None
            ),
            schedule, None, None, jax.random.PRNGKey(0), None, None, resume_epoch=3,
        )
    assert seen == [0, 0, 1, 1, 2, 2]


def test_run_training_hands_the_sampler_the_same_keys_it_would_have_had():
    """Replaying the splits leaves the RNG where an uninterrupted run would have it."""
    schedule = Schedule.make(
        num_timesteps=64, num_evals=4, num_envs=8, batch_size=4,
        num_minibatches=2, unroll_length=2, resamples_per_epoch=1,
    )

    def keys_from(resume_epoch):
        """The sampler keys the replay hands a job that starts at ``resume_epoch``."""
        seen = []

        def sample(it, extra_state, key):
            seen.append(np.asarray(jax.random.key_data(key)))
            return None, None

        # Stop once enough draws have been seen, rather than running the whole loop.
        def evaluate(*args):
            raise _Stop

        with pytest.raises(_Stop):
            shared.run_training(
                shared.Algorithm(
                    sample=sample, chunk=None, evaluate=evaluate, params_of=None
                ),
                schedule, None, None, jax.random.PRNGKey(0), None, None,
                resume_epoch=resume_epoch,
            )
        return seen

    # Resuming at epoch 3 redraws exactly the keys the first three epochs used, so the
    # fourth epoch continues the stream rather than repeating the first.
    assert len(keys_from(3)) == 3
    assert np.array_equal(keys_from(2), keys_from(3)[:2])


# ------------------------------------------------------------------------- end to end

def _resume_config(case, tmp_path, num_timesteps):
    config = mm.deepcopy_config(case("design_hypernetwork")[0])
    config.save_dir = str(tmp_path)
    config.name = "resumed"
    config.learning_params.ppo_params.update({**TINY, "num_timesteps": num_timesteps})
    config.learning_params.design_sampling.update(dict(
        num_designs=2, per_cell=4, strategy="noisy-fixed", warmup_strategy="fixed",
        warmup_epochs=1, rate=0.1, resamples_per_epoch=1,
    ))
    config.learning_params.network_params.initialization_strategy = "bias"
    if "warmup_checkpoint" in config.learning_params.network_params:
        del config.learning_params.network_params["warmup_checkpoint"]
    return config


@pytest.mark.slow
def test_a_continued_run_extends_the_same_directory(case, tmp_path):
    """Raising num_timesteps and resuming trains the difference, in place."""
    env = case("design_hypernetwork")[1]
    train = lambda config, resume: mm.learning.training.train(
        config, env, env,
        handle_params=codesign.setup_design_hypernetwork, resume=resume,
    )
    run_dir = tmp_path / "resumed"
    steps = TINY["num_timesteps"]

    train(_resume_config(case, tmp_path, steps), resume=False)
    first = [p.name for p in mm.find_checkpoints(run_dir)]
    assert first == ["000000000000", "000000000016", "000000000032"]
    assert (run_dir / "training_state").exists()
    # The continued job re-checkpoints the state it restarts from, over this one.
    stopped_on = brax_checkpoint.load(str(run_dir / f"{steps:012d}"))

    train(_resume_config(case, tmp_path, 2 * steps), resume=True)

    second = [p.name for p in mm.find_checkpoints(run_dir)]
    # The steps carry on from where the first job stopped, and nothing is overwritten.
    assert second == first + ["000000000048", "000000000064"]
    # config.yaml records the whole budget, not the part this job was handed.
    assert mm.read_config(run_dir / "config.yaml").learning_params.ppo_params \
        .num_timesteps == 2 * steps
    # One progress row per eval, the row at the seam being this job's re-evaluation.
    assert pd.read_csv(run_dir / "progress.csv")["x"].tolist() == [0, 16, 32, 48, 64]
    # The saved learner state put the params back exactly, so what the second job
    # checkpointed at the seam is what the first job stopped on.
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(a, b),
        stopped_on, brax_checkpoint.load(str(run_dir / f"{steps:012d}")),
    )
