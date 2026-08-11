"""Rolls outs codesign hypernetworks given a config yaml.
"""
import os

from codesign.envs.codesign_base import CodesignMO2SO
os.environ["MUJOCO_GL"] = "egl"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
from pathlib import Path

import numpy as np
import minimal_mjx as mm
import matplotlib.pyplot as plt
import moplayground as mop
import codesign

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"
OUT_DIR = Path("scripts/outputs")


def resolve_design(values: list[float] | None, config) -> np.ndarray:
    """CLI design values -> a ``(design_dim,)`` array.

    ``None`` falls back to the config's default design; a single value is broadcast
    across every design dimension.
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


def main(
    config_path: str,
    checkpoint_path: str | None,
    design: list[float] | None,
    eval_design: list[float] | None,
    tradeoff: list[float] | None,
    design_tradeoff: list[float] | None,
    sample_design: bool,
    steps: int,
    camera: str
) -> None:
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env, env_params = codesign.load_env(config, backend = 'np')

    # Keep the raw CLI values: the design predictor needs to tell "not given" (use its
    # own prediction) from "defaulted to the config's design".
    cli_design, cli_eval_design = design, eval_design
    design      = resolve_design(design, config)                                  # (design_dim,)
    eval_design = design if eval_design is None else resolve_design(eval_design, config)
    algorithm = config["algorithm"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if algorithm == "mo_design_hypernetwork":
        opt         = config["env_config"]["reward"]["optimization"]
        objectives  = codesign.utils.plotting.objective_labels(opt["objectives"])
        num_obj     = len(objectives)

        if tradeoff is None:
            tradeoff = [1.0 / num_obj] * num_obj
        norm = [w / sum(tradeoff) for w in tradeoff]

        print("objective scalarization (raw -> normalized):")
        for obj, raw, w in zip(objectives, tradeoff, norm):
            print(f"  {obj}: {raw:g} -> {w:.3f}")

        # Rollout
        frames, traj, reward_plotter, _, _ = codesign.rollout_mo_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, tradeoff=tradeoff, n_steps=steps,
            checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
        )
        out = OUT_DIR / f"mo_design_hypernetwork.mp4"
        title = f"{config['env']} d={np.round(design, 3)} w={np.round(tradeoff, 3)} reward"

    elif algorithm == "mo_design_predictor_hypernetwork":
        opt         = config["env_config"]["reward"]["optimization"]
        objectives  = codesign.utils.plotting.objective_labels(opt["objectives"])
        num_obj     = len(objectives)

        if tradeoff is None:
            tradeoff = [1.0 / num_obj] * num_obj
        # The predictor's tradeoff may differ from the policy's, to test specificity.
        w_design = tradeoff if design_tradeoff is None else design_tradeoff

        # The predictor supplies the design, so the model is built from its prediction
        # unless the CLI names one to override it (--eval_design wins over --design).
        override = cli_eval_design if cli_eval_design is not None else cli_design
        override = None if override is None else resolve_design(override, config)

        # Rollout
        frames, traj, reward_plotter, _, _, design = (
            codesign.rollout_mo_design_predictor_hypernetwork_video(
                env, config, tradeoff=tradeoff, design_tradeoff=w_design,
                eval_design=override, sample_design=sample_design, n_steps=steps,
                checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
            )
        )
        eval_design = design if override is None else override
        print(f"predicted design f(. | w'={np.round(w_design, 3)}) = {np.round(design, 3)}")
        out = OUT_DIR / f"mo_design_predictor_hypernetwork.mp4"
        title = (
            f"{config['env']} d={np.round(design, 3)} "
            f"w={np.round(tradeoff, 3)} w'={np.round(w_design, 3)} reward"
        )

    elif algorithm == "design_hypernetwork":
        if tradeoff is not None:
            print("note: H(d) is single-objective; --tradeoff is ignored.")

        # Rollout
        frames, traj, reward_plotter, _, _ = codesign.rollout_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, n_steps=steps,
            checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
        )
        out = OUT_DIR / f"design_hypernetwork.mp4"
        title = f"{config['env']} d={np.round(design, 3)} reward"

    elif algorithm == "ppo":
        # Fixed-design policy: the design only picks the model the rollout is run on.
        base_policy = mm.load_policy(config, deterministic=True, checkpoint_path=checkpoint_path)
        policy      = mm.from_inference_fn(base_policy)
        so_env      = (
            env if isinstance(env, CodesignMO2SO)
            else CodesignMO2SO(env, config["env_config"]["reward"]["optimization"]["default_scalarization"])
        )
        # Rollout
        frames, traj, reward_plotter, _, _ = codesign.rollout_single_video(
            so_env, eval_design, policy, steps,
            camera=camera, width=640, height=480,
        )
        out = OUT_DIR / f"ppo.mp4"
        title = f"{config['env']} d={np.round(eval_design, 3)} reward"


    else:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; expected 'ppo', 'design_hypernetwork', "
            "'mo_design_hypernetwork' or 'mo_design_predictor_hypernetwork'."
        )

    mm.utils.plotting.save_video(frames, env.dt, out)
    print(f"rendered {len(traj)} steps for design d={np.round(design, 3)} (eval d={np.round(eval_design, 3)}).")

    reward_plotter.plot(title=title)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"rendered plots -> {out.with_suffix(".pdf")}")
    discount = config.learning_params.ppo_params.discounting
    rewards  = np.asarray(reward_plotter.rewards)
    # MO envs carry a trailing objective axis, so broadcast the discount along time only.
    discounts = np.pow(discount, np.arange(len(rewards))).reshape(
        -1, *([1] * (rewards.ndim - 1))
    )
    print(f"Total value: {np.sum(rewards, axis=0)}")
    print(f"Discounted value: {np.sum(rewards * discounts, axis=0)}")


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
        eval_design = args.eval_design
    )
