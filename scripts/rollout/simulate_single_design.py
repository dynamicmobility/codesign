"""Simulate an open-loop policy on a single design.
"""
import os

import codesign
from codesign.envs.EnvLoader import load_env
os.environ["MUJOCO_GL"] = "egl"

import argparse
from pathlib import Path

import minimal_mjx as mm
from minimal_mjx.eval import make_open_loop_policy
from codesign.eval.rollout_video import rollout_single_video
from codesign.utils.model import total_mass
from matplotlib import pyplot as plt
import numpy as np

CONFIG_PATH = "config/design_hypernetwork_two_axis.yaml"

T = 250          # rollout length (control steps)
AMP = 0.8        # action amplitude (ctrl range is [-1, 1])
FREQ = 0.5       # action frequency [Hz]
OUT_DIR = Path("scripts/outputs")


def main(config: str, design: float | np.ndarray | None, steps: int, policy_kind: str, camera: str) -> None:


    train_config = mm.utils.read_config(config)
    env_params = mm.utils.config.create_config_dict(train_config["env_config"])
    env, _ = load_env(config=train_config, backend="jnp")
    
    if(train_config["env"] == "TwoAxis"):
        def pd(obs, key, t):
            K = np.array([[-10, 0, -10, 0], [0, -10, 0, -10]])
            frc = K @ (obs - np.array([2.0, 2.0, 0.0, 0.0]))
            return frc, {}
        policy = pd
    elif(train_config["env"] == "RHex"):
        def my_policy(obs, key, t):
            frc = -1.0*np.ones((6,))
            return frc, {}
        policy = my_policy
    else:
        policy = make_open_loop_policy(policy_kind, env.action_size, amp=AMP, freq=FREQ)

    if(design is None):
        design = env.default_design
        print(design)
    frames, traj, reward_plotter, data_plotter, info_plotter = rollout_single_video(
        env, design, policy, steps, camera=camera, width=640, height=480,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{train_config['env']}_d{str(design).replace('.', '_')}_{policy_kind}.mp4"
    mm.utils.plotting.save_video(frames, env.dt, out)
    print(f"rendered {len(traj)} steps ({policy_kind} policy, d={design}) -> {out}")

    reward_plotter.plot(title=f"{train_config['env']} d={design} reward")

    if(train_config["env"] == "RHex"):
        env: codesign.envs.RHex = env
        qpos_list = []
        for step in traj:
            qpos_list.append(step.data.qpos.copy())

        qpos_arr = np.stack(qpos_list)
        joint_ids = env.actuated_joint_pos_idxs()

        plt.figure()
        for jid in np.asarray(joint_ids).ravel():
            plt.plot(qpos_arr[:, jid], label=f"joint {jid}")
        plt.title(f"{train_config['env']} d={design} actuated joint positions")
        plt.xlabel("step")
        plt.ylabel("position")
        plt.legend()


    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=float, default=None, help="design vector")
    parser.add_argument("--steps", type=int, default=T, help="rollout length (control steps)")
    parser.add_argument(
        "--policy", type=str, default="sinusoid",
        choices=["sinusoid", "random", "zero"], help="open-loop action policy",
    )
    parser.add_argument("--camera", type=str, default="track", help="render camera name")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="environment config to use")
    args = parser.parse_args()
    main(args.config, args.design, args.steps, args.policy, args.camera)
