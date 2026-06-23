"""Record a trained ``design_hypernetwork`` policy on a single design.
"""

import argparse
from pathlib import Path

import minimal_mjx as mm
import moplayground as mop
from codesign.envs.MAICheetah import MAICheetah
from codesign.eval import rollout_design_hypernetwork_video

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"

D = 2.0          # back-leg length scale to render
T = 500          # rollout length (env steps)
OUT_DIR = Path("scripts/outputs")


def main(config_path: str, checkpoint_path: str | None, design: float,
         steps: int, camera: str) -> None:
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env = MAICheetah(env_params=env_params, backend="np")

    frames, traj = rollout_design_hypernetwork_video(
        env, config, design=design, n_steps=steps,
        checkpoint_path=checkpoint_path, camera=camera, width=640, height=480,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"design_hypernetwork_d{str(design).replace('.', '_')}.mp4"
    mm.utils.plotting.save_video(frames, env.dt, out)
    print(f"rendered {len(traj)} steps for design d={design} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="explicit checkpoint dir; defaults to latest under save_dir/name",
    )
    parser.add_argument("--design", type=float, default=D, help="back-leg length scale")
    parser.add_argument("--steps", type=int, default=T, help="rollout length (env steps)")
    parser.add_argument("--camera", type=str, default="track", help="render camera name")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.design, args.steps, args.camera)
