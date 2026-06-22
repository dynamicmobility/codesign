"""Simulate an open-loop policy on a single MAICheetah design.
"""

import argparse
from pathlib import Path

import minimal_mjx as mm
import moplayground as mop
from codesign.envs.MAICheetah import MAICheetah
from codesign.eval import rollout_single, make_open_loop_policy

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"

D = 1.0          # back-leg length scale to render
T = 50           # rollout length (control steps)
AMP = 0.8        # action amplitude (ctrl range is [-1, 1])
FREQ = 1.5       # action frequency [Hz]
OUT_DIR = Path("scripts/outputs")


def main(design: float, steps: int, policy_kind: str, camera: str) -> None:
    train_config = mop.utils.read_config(CONFIG_PATH)
    env_params = mm.utils.config.create_config_dict(train_config["env_config"])
    env = MAICheetah(env_params=env_params, backend="jnp")

    policy = make_open_loop_policy(policy_kind, env.action_size, amp=AMP, freq=FREQ)
    frames, traj = rollout_single(
        env, design, policy, steps, camera=camera, width=640, height=480,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mai_cheetah_d{str(design).replace('.', '_')}_{policy_kind}.mp4"
    mm.utils.plotting.save_video(frames, env.dt, out)
    print(f"rendered {len(traj)} steps ({policy_kind} policy, d={design}) -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=float, default=D, help="back-leg length scale")
    parser.add_argument("--steps", type=int, default=T, help="rollout length (control steps)")
    parser.add_argument(
        "--policy", type=str, default="sinusoid",
        choices=["sinusoid", "random", "zero"], help="open-loop action policy",
    )
    parser.add_argument("--camera", type=str, default="track", help="render camera name")
    args = parser.parse_args()
    main(args.design, args.steps, args.policy, args.camera)
