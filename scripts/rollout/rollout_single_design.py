"""Rolls outs codesign hypernetworks given a config yaml.
"""
import os

from codesign.envs.CodesignBase import CodesignMO2SO
os.environ["MUJOCO_GL"] = "egl"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
from pathlib import Path

import minimal_mjx as mm
import matplotlib.pyplot as plt
import moplayground as mop
from codesign.eval.rollout_video import rollout_design_hypernetwork_video, rollout_mo_design_hypernetwork_video, rollout_single_video
from codesign.envs.EnvLoader import load_env

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"
OUT_DIR = Path("scripts/outputs")


def main(
    config_path: str, 
    checkpoint_path: str | None, 
    design: float,
    eval_design: float | None,
    tradeoff: list[float] | None, 
    steps: int, 
    camera: str
) -> None:
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env, env_params = load_env(config, backend = 'np')

    algorithm = config["algorithm"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if algorithm == "mo_design_hypernetwork":
        opt         = config["env_config"]["reward"]["optimization"]
        objectives  = list(opt["objectives"])
        labels      = list(opt["objectives"])
        num_obj     = len(objectives)
        
        if tradeoff is None:
            tradeoff = [1.0 / num_obj] * num_obj
        norm = [w / sum(tradeoff) for w in tradeoff]
        
        print("objective scalarization (raw -> normalized):")
        for label, obj, raw, w in zip(labels, objectives, tradeoff, norm):
            print(f"  {label} {obj}: {raw:g} -> {w:.3f}")
        
        # Rollout
        frames, traj, reward_plotter, _, _ = rollout_mo_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, tradeoff=tradeoff, n_steps=steps,
            checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
        )
        out = OUT_DIR / f"mo_design_hypernetwork.mp4"
        title = f"{config['env']} d={design} w={tradeoff} reward"
        
    elif algorithm == "design_hypernetwork":
        if tradeoff is not None:
            print("note: H(d) is single-objective; --tradeoff is ignored.")
        
        # Rollout
        frames, traj, reward_plotter, _, _ = rollout_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, n_steps=steps,
            checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
        )
        out = OUT_DIR / f"design_hypernetwork.mp4"
        title = f"{config['env']} d={design} reward"
    
    elif algorithm == "ppo":
        # Trained, design-conditioned policy for this single design (1-D design -> unbatched).
        policy = mm.learning.inference.load_policy(config)
        so_eval_env = CodesignMO2SO(env, config['learning_params']['reward_objective_weights'])
        # Rollout
        frames, traj, reward_plotter, _, _ = mm.eval.rollout_policy(
            inference_fn=policy,
            env= so_eval_env, 
            n_steps=steps,
            camera=camera, 
            width=640, height=480,
        )
        out = OUT_DIR / f"design_hypernetwork.mp4"
        title = f"{config['env']} d={design} reward"

        
    else:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; expected 'design_hypernetwork' or "
            "'mo_design_hypernetwork'."
        )

    mm.utils.plotting.save_video(frames, env.dt, out)
    print(f"rendered {len(traj)} steps for design d={design}.")
    
    reward_plotter.plot(title=title)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"rendered plots -> {out.with_suffix(".pdf")}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="explicit checkpoint dir; defaults to latest under save_dir/name",
    )
    parser.add_argument("--design", type=float, default=1.0, help="design parameter for hypernet input")
    parser.add_argument(
        "--tradeoff", type=float, nargs="+", default=None,
        help="objective scalarization w. Only used for mo_design_hypernetwork.",
    )
    parser.add_argument("--steps", type=int, default=500, help="rollout length (env steps)")
    parser.add_argument("--camera", type=str, default="track", help="render camera name")
    parser.add_argument("--eval_design", type=float, default = 1.0, help="design parameter for environment evaluation")
    args = parser.parse_args()
    main(
        config_path = args.config, 
        checkpoint_path = args.checkpoint, 
        design = args.design, 
        tradeoff = args.tradeoff, 
        steps = args.steps, 
        camera = args.camera, 
        eval_design = args.eval_design
    )
