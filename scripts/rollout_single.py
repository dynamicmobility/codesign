"""Rolls outs codesign hypernetworks given a config yaml.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import minimal_mjx as mm
import matplotlib.pyplot as plt
import moplayground as mop
import codesign
from codesign.utils.model import normalize_design

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"
OUT_DIR = Path("scripts/outputs")

MO_ALGORITHMS = ("mo_design_hypernetwork", "mo_design_predictor_hypernetwork", "mo_design_mlp")


def resolve_single_design(values: list[float] | None, config) -> np.ndarray:
    """CLI design values -> a ``(design_dim,)`` array.

    ``None`` falls back to the config's default design
    """
    if values is None:
        return codesign.default_video_design(config)

    dim = len(np.atleast_1d(config["env_config"]["codesign"]["low"]))
    design = np.asarray(values, np.float32).reshape(-1)
    if design.size == 1:
        design = np.full((dim,), design[0], np.float32)
    if design.size != dim:
        raise ValueError(f"design has {design.size} entries; env expects {dim}")
    return design


def resolve_design_args(
    design: list[float] | None, eval_design: list[float] | None, config
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """CLI designs -> the hypernetwork's design and the design the model is built from.

    The design predictor supplies its own design, so nothing is defaulted for it: only an
    explicit CLI value overrides the model's design, and ``--eval_design`` wins over
    ``--design``.
    """
    if config["algorithm"] == "mo_design_predictor_hypernetwork":
        override = eval_design if eval_design is not None else design
        return None, (None if override is None else resolve_single_design(override, config))

    design = resolve_single_design(design, config)
    if eval_design is None:
        return design, design
    return design, resolve_single_design(eval_design, config)


def normalize_objectives(values: list[float], objectives: list[str], label: str) -> np.ndarray:
    """Weights normalized onto the simplex, printed per objective."""
    raw = np.asarray(values, np.float32).reshape(-1)
    if raw.size != len(objectives):
        raise ValueError(f"{label} has {raw.size} entries; env has {len(objectives)} objectives")

    w = raw / np.sum(raw)
    print(f"objective scalarization {label} (raw -> normalized):")
    for obj, r, norm in zip(objectives, raw, w):
        print(f"  {obj}: {r:g} -> {norm:.3f}")
    return w


def resolve_tradeoffs_args(
    tradeoff: list[float] | None, design_tradeoff: list[float] | None, config
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """CLI tradeoffs -> the policy's ``w`` and the design predictor's ``w'``.

    Only the MO hypernetworks condition on a tradeoff, so both are dropped elsewhere.
    ``w`` defaults to uniform; ``w'`` stays ``None`` so it falls back to ``w`` downstream.
    """
    if config["algorithm"] not in MO_ALGORITHMS:
        if tradeoff is not None or design_tradeoff is not None:
            print(f"note: {config['algorithm']} is single-objective; tradeoffs are ignored.")
        return None, None

    opt        = config["env_config"]["reward"]["optimization"]
    objectives = codesign.utils.plotting.objective_labels(opt["objectives"])
    if tradeoff is None:
        tradeoff = [1.0 / len(objectives)] * len(objectives)

    tradeoff = normalize_objectives(tradeoff, objectives, "w")
    if design_tradeoff is not None:
        design_tradeoff = normalize_objectives(design_tradeoff, objectives, "w'")
    return tradeoff, design_tradeoff


def save_reward_plot(rollout: codesign.RolloutVideo) -> Path:
    """Write the rollout's per-objective reward traces next to its video."""
    out = rollout.path.with_suffix(".pdf")
    rollout.reward_plotter.plot(title=f"{rollout.caption} reward")
    plt.savefig(out)
    return out


def report_features(config, design, checkpoint_path: str | None) -> None:
    """Print the hypernetwork's feature row for ``design``, if it has one.

    ``flat(d) = features(d) @ W + b``, so the row is the mix of the ``num_features``
    experts this policy is built from: exactly one-hot for the lookup, and for a full
    hypernetwork whatever its MLP learned -- near one-hot at a warm-up anchor if the
    transfer took, spread out otherwise. Nothing constrains the MLP's output to be
    positive, and a large negative weight subtracts an expert just as strongly as a
    positive one adds it, so entries are ranked by magnitude rather than by value.
    """
    features, params = codesign.load_design_features(config, path=checkpoint_path)
    if features is None:
        return

    row = np.asarray(
        features(params[1], normalize_design(jnp.asarray(design), config=config))
    ).reshape(-1)
    order = np.argsort(np.abs(row))[::-1][:5]
    print(f"feature row ({row.size} experts), largest magnitude first:")
    for i in order:
        print(f"  expert {i:3d}: {row[i]:+.4f}")
    print(f"  sum {row.sum():+.4f}, argmax {int(np.argmax(row))}, "
          f"min {row.min():+.4f} (expert {int(np.argmin(row))}), max {row.max():+.4f}")
    print(f"  total abs mass {np.abs(row).sum():.4f}, "
          f"outside the {len(order)} listed {np.abs(row).sum() - np.abs(row[order]).sum():.4f}")


def report_value(rollout: codesign.RolloutVideo, config) -> None:
    """Print the rollout's undiscounted and discounted return, per objective."""
    discount = config.learning_params.ppo_params.discounting
    rewards  = np.asarray(rollout.reward_plotter.rewards)
    # MO envs carry a trailing objective axis, so broadcast the discount along time only.
    discounts = np.pow(discount, np.arange(len(rewards))).reshape(
        -1, *([1] * (rewards.ndim - 1))
    )
    print(f"Total value: {np.sum(rewards, axis=0)}")
    print(f"Discounted value: {np.sum(rewards * discounts, axis=0)}")


def main(
    config_path: str,
    checkpoint_path: str | None,
    design: list[float] | None,
    eval_design: list[float] | None,
    tradeoff: list[float] | None,
    design_tradeoff: list[float] | None,
    sample_design: bool,
    steps: int,
    camera: str,
    use_caption: bool
) -> None:
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env, _ = codesign.load_env(config, backend = 'jnp')  # renderable (mujoco) env

    design, eval_design       = resolve_design_args(design, eval_design, config)
    tradeoff, design_tradeoff = resolve_tradeoffs_args(tradeoff, design_tradeoff, config)

    report_features(config, design if design is not None else eval_design, checkpoint_path)

    rollout = codesign.save_policy_rollout_video(
        config,
        OUT_DIR / f"{config['algorithm']}.mp4",
        env             = env,
        design          = design,
        eval_design     = eval_design,
        tradeoff        = tradeoff,
        design_tradeoff = design_tradeoff,
        sample_design   = sample_design,
        n_steps         = steps,
        checkpoint_path = checkpoint_path,
        camera          = camera,
        width           = 640,
        height          = 480,
        use_caption     = use_caption,
        deterministic   = True,
        seed            = 21,
    )
    print(f"rendered video -> {rollout.path}")
    print(f"rendered plots -> {save_reward_plot(rollout)}")
    report_value(rollout, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="explicit checkpoint dir; defaults to latest under save_dir/name",
    )
    parser.add_argument(
        "--design", type=float, nargs="+", default=None,
        help="design vector for hypernet input; a single value is broadcast across all "
             "design dimensions. defaults to the config's default design",
    )
    parser.add_argument(
        "--tradeoff", type=float, nargs="+", default=None,
        help="objective scalarization w conditioning the policy. Only used for the MO "
             "hypernetworks.",
    )
    parser.add_argument(
        "--design_tradeoff", type=float, nargs="+", default=None,
        help="tradeoff w' fed to the design predictor f(d | w'); defaults to --tradeoff. "
             "Only used for mo_design_predictor_hypernetwork",
    )
    parser.add_argument(
        "--sample_design", action="store_true",
        help="sample from f(d | w') instead of taking its mode",
    )
    parser.add_argument("--steps", type=int, default=500, help="rollout length (env steps)")
    parser.add_argument("--camera", type=str, default="track", help="render camera name")
    parser.add_argument(
        "--eval_design", type=float, nargs="+", default=None,
        help="design vector the environment model is built from. defaults to design",
    )

    parser.add_argument("--caption", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(
        config_path = args.config,
        checkpoint_path = args.checkpoint,
        design = args.design,
        tradeoff = args.tradeoff,
        design_tradeoff = args.design_tradeoff,
        sample_design = args.sample_design,
        steps = args.steps,
        camera = args.camera,
        eval_design = args.eval_design,
        use_caption = args.caption
    )
