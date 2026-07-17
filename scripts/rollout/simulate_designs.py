"""Roll out N CodesignCheetah variants with different back-leg lengths in parallel.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mujoco import mjx

import minimal_mjx as mm
import codesign
from codesign.envs.CodesignCheetah import CodesignCheetah
from codesign.eval import rollout_parallel
from codesign.utils.model import build_batched_model

N = 256          # number of cheetah variants
T = 50           # rollout length (control steps)
AMP = 0.8        # action amplitude (ctrl range is [-1, 1])
FREQ = 1.5       # action frequency [Hz]
D_MIN = 0.5      # min back-leg length scale
D_MAX = 2.0      # max back-leg length scale
OUT_DIR = Path('scripts/outputs')


def main(config: str, n: int, steps: int, policy_desc: str) -> None:
    train_config = mm.utils.read_config(config)
    env, env_params = codesign.envs.EnvLoader.load_env(train_config)

    # 1. sample leg-length scales and build/stack the batched model
    ds = np.linspace(D_MIN, D_MAX, n)
    batched_model = build_batched_model(env, ds.reshape(n, 1))

    # 2. make a dummy policy
    policy = mm.make_open_loop_policy(policy_desc, env.action_size, amp=AMP, freq=FREQ)

    # 3. parallel rollout, recording base [x, z] each step (no termination masking)
    record_xz = lambda prev, act, nst: nst.data.qpos[:, jnp.array([0, 1])]
    traj = rollout_parallel(
        env, batched_model, policy, n, steps,
        record_fn=record_xz, mask_after_done=False,
    )  # (T, N, 2)
    time = np.arange(steps) * env.dt

    # 4. save trajectories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f'{train_config["env"]}_parallel_simulate.npz', time=time, xpos=traj, ds=ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, help='config')
    parser.add_argument('--n', type=int, default=N, help='number of envs')
    parser.add_argument('--steps', type=int, default=T, help='rollout length (control steps)')
    parser.add_argument(
        '--policy', type=str, default='sinusoid',
        choices=['sinusoid', 'random', 'zero'], help='open-loop action policy',
    )
    args = parser.parse_args()
    main(args.config, args.n, args.steps, args.policy)
