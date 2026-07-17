"""Roll out N CodesignCheetah variants with different back-leg lengths in parallel.
Several 'open-loop' policies are available (sinusoid, random, zero).
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mujoco import mjx

import minimal_mjx as mm
from codesign.envs.CodesignCheetah import CodesignCheetah
from codesign.eval import rollout_parallel, make_open_loop_policy
from codesign.utils.model import build_batched_model

CONFIG_PATH = 'config/design_hypernetwork_cheetah.yaml'

N = 256          # number of cheetah variants
T = 50           # rollout length (control steps)
AMP = 0.8        # action amplitude (ctrl range is [-1, 1])
FREQ = 1.5       # action frequency [Hz]
D_MIN = 0.5      # min back-leg length scale
D_MAX = 2.0      # max back-leg length scale
OUT_DIR = Path('scripts/outputs')


def main(n: int, steps: int, policy_kind: str) -> None:
    train_config = mm.utils.read_config(CONFIG_PATH)
    env_params = mm.utils.config.create_config_dict(train_config['env_config'])
    env = CodesignCheetah(env_params=env_params, backend='jnp')

    # 1. sample leg-length scales and build/stack the batched model
    ds = np.linspace(D_MIN, D_MAX, n)

    def generate_model_fn(design_row):
        d = float(np.asarray(design_row).reshape(-1)[0])
        return mjx.put_model(env.generate_model(d))

    batched_model = build_batched_model(generate_model_fn, ds.reshape(n, 1))

    # 2. open-loop policy (sinusoid / random / zero), identical across all variants
    policy = make_open_loop_policy(policy_kind, env.action_size, amp=AMP, freq=FREQ)

    # 3. parallel rollout, recording base [x, z] each step (no termination masking)
    record_xz = lambda prev, act, nst: nst.data.qpos[:, jnp.array([0, 1])]
    traj = rollout_parallel(
        env, batched_model, policy, n, steps,
        record_fn=record_xz, mask_after_done=False,
    )  # (T, N, 2)
    time = np.arange(steps) * env.dt

    # 4. save trajectories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / 'mai_cheetah_leglength_traj.npz', time=time, xpos=traj, ds=ds)

    # 5. plot x and z vs time (one translucent line per model)
    fig, (ax_x, ax_z) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_x.plot(time, traj[:, :, 0], color='C0', alpha=0.1)
    ax_z.plot(time, traj[:, :, 1], color='C0', alpha=0.1)
    ax_x.set_ylabel('base x [m]')
    ax_z.set_ylabel('base z [m]')
    ax_z.set_xlabel('time [s]')
    ax_x.set_title(
        f'{n} CodesignCheetahs ({policy_kind} policy), back-leg scaled in [{D_MIN}, {D_MAX}]'
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'codesign_cheetah_leglength_xz.png', dpi=150)
    plt.close(fig)

    print(f'traj shape {traj.shape} ({policy_kind} policy) -> {OUT_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=N, help='number of cheetah variants')
    parser.add_argument('--steps', type=int, default=T, help='rollout length (control steps)')
    parser.add_argument(
        '--policy', type=str, default='sinusoid',
        choices=['sinusoid', 'random', 'zero'], help='open-loop action policy',
    )
    args = parser.parse_args()
    main(args.n, args.steps, args.policy)
