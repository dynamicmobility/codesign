"""Evaluate a trained ``design_hypernetwork`` across a uniform sweep of designs.

Loads a design-conditioned hypernetwork checkpoint (the one minimal-mjx wrote during
``scripts/train_design_hypernetwork.py``), reconstructs the policy via the same
network factory, then rolls a uniform sweep of robot designs (back-leg length scales
``d`` in ``[design_low, design_high]``) out for ``--steps`` env steps. The per-design,
per-step scalar reward is logged and turned into two figures:

  1. reward vs. time, one translucent line per design (coloured by design value);
  2. a bar chart of cumulative reward per design.

The checkpoint, networks and design scalarization are all reconstructed from the run's
``config.yaml`` so this matches the training-time setup. Structurally a sibling of
``scripts/rollout_mai_cheetah.py`` (same ``stack_models`` batching), but it drives the
*trained* hypernet policy rather than a fixed sinusoid.

Runs on GPU by default. Force CPU with:
    JAX_PLATFORMS=cpu python scripts/rollout_design_hypernetwork.py
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from brax.training import checkpoint
from brax.training.acme import running_statistics

import minimal_mjx as mm
import moplayground as mop
from codesign.envs.MAICheetah import MAICheetah
from codesign.hyperdesigners import acting
from codesign.hyperdesigners import models as model_lib
from codesign.hyperdesigners.networks import make_design_inference_fn
from codesign.hyperdesigners.factory import setup_design_hypernetwork

CONFIG_PATH = "config/design_hypernetwork_cheetah.yaml"

N = 64           # number of designs in the uniform sweep
T = 500          # rollout length (env steps)
OUT_DIR = Path("scripts/outputs")


def find_latest_checkpoint(run_dir: Path) -> Path:
    """Return the highest-step checkpoint subdirectory written under ``run_dir``."""
    steps = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not steps:
        raise FileNotFoundError(f"no step checkpoints found under {run_dir}")
    return steps[-1]


def main(config_path: str, checkpoint_path: str | None, n: int, steps: int) -> None:
    # 1. Load the run config and rebuild the env + network factory exactly as in training.
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env_params = mm.utils.config.create_config_dict(config["env_config"])
    env = MAICheetah(env_params=env_params, backend="jnp")

    design = config["learning_params"]["design_params"]
    design_low = float(design["design_low"])
    design_high = float(design["design_high"])
    design_dim = int(design["design_dim"])

    _, network_factory = setup_design_hypernetwork(config)

    # 2. Locate and load the checkpoint params: (normalizer_params, hypernet_params).
    if checkpoint_path is None:
        run_dir = Path(config["save_dir"]) / config["name"]
        ckpt = find_latest_checkpoint(run_dir)
    else:
        ckpt = Path(checkpoint_path)
    print(f"loading checkpoint: {ckpt}")
    normalizer_params, hypernet_params = checkpoint.load(ckpt.resolve())

    # 3. Build a uniform sweep of designs and the stacked, batched mjx.Model.
    ds = np.linspace(design_low, design_high, n).reshape(n, design_dim).astype(np.float32)

    def generate_model_fn(design_row):
        d = float(np.asarray(design_row).reshape(-1)[0])
        from mujoco import mjx
        return mjx.put_model(env.generate_model(d))

    batched_model = model_lib.build_batched_model(generate_model_fn, ds)
    designs_input = model_lib.normalize_design(jnp.asarray(ds), design_low, design_high)

    # 4. Rebuild the networks (preprocess with the saved normalizer) + inference fn.
    rngs = jax.random.split(jax.random.PRNGKey(0), n)
    env_state = acting.reset(env, rngs, batched_model)
    obs_size = jax.tree_util.tree_map(lambda x: x.shape[-1], env_state.obs)

    normalize = running_statistics.normalize if config["learning_params"][
        "ppo_params"
    ]["normalize_observations"] else (lambda x, y: x)
    design_networks = network_factory(
        observation_size=obs_size,
        action_size=env.action_size,
        design_dim=design_dim,
        key=jax.random.PRNGKey(0),
        preprocess_observations_fn=normalize,
    )
    inference_fn = make_design_inference_fn(design_networks)

    # 5. Scalarize MAICheetah's multi-objective reward exactly as training did.
    num_objectives = int(env_state.reward.shape[-1])
    weights = config["learning_params"].get("reward_objective_weights")
    reward_weights = (
        jnp.ones(num_objectives) if weights is None else jnp.asarray(weights, jnp.float32)
    )
    scalarize_reward = lambda r: jnp.sum(r * reward_weights, axis=-1)

    # 6. Deterministic rollout of all designs in parallel, logging per-step reward.
    @jax.jit
    def rollout(rngs, key):
        state = acting.reset(env, rngs, batched_model)
        policy = inference_fn(
            (normalizer_params, hypernet_params), designs_input, deterministic=True
        )

        def body(carry, _):
            st, k, alive = carry
            k, sub = jax.random.split(k)
            act, _ = policy(st.obs, sub)
            nst = jax.vmap(env.step, in_axes=(0, 0, 0))(st, act, batched_model)
            step_reward = scalarize_reward(nst.reward) * alive   # zero after termination
            alive = alive * (1.0 - nst.done)
            return (nst, k, alive), step_reward

        init = (state, key, jnp.ones(n))
        _, rewards = jax.lax.scan(body, init, (), length=steps)
        return rewards   # (T, N)

    rng_reset = jax.random.split(jax.random.PRNGKey(1), n)
    rewards = np.asarray(rollout(rng_reset, jax.random.PRNGKey(2)))   # (T, N)
    cumulative = rewards.sum(axis=0)                                  # (N,)
    time = np.arange(steps)

    # 7. Save logged rewards.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "design_hypernetwork_rewards.npz",
        time=time, rewards=rewards, cumulative=cumulative, designs=ds,
    )

    # 8a. Reward vs. time, one line per design (coloured by design value).
    norm = Normalize(vmin=design_low, vmax=design_high)
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n):
        ax.plot(time, rewards[:, i], color=cmap(norm(ds[i, 0])), alpha=0.4, lw=0.8)
    ax.set_xlabel("env step")
    ax.set_ylabel("scalar reward")
    ax.set_title(f"design_hypernetwork reward over time ({n} designs, {steps} steps)")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="design d (back-leg length scale)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "design_hypernetwork_reward_over_time.png", dpi=150)
    plt.close(fig)

    # 8b. Cumulative reward per design (bar chart).
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [cmap(norm(d)) for d in ds[:, 0]]
    width = (design_high - design_low) / n * 0.9
    ax.bar(ds[:, 0], cumulative, width=width, color=bar_colors)
    ax.set_xlabel("design d (back-leg length scale)")
    ax.set_ylabel(f"cumulative reward over {steps} steps")
    ax.set_title(f"design_hypernetwork cumulative reward by design ({n} designs)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "design_hypernetwork_cumulative_reward.png", dpi=150)
    plt.close(fig)

    print(
        f"rewards {rewards.shape} | cumulative mean {cumulative.mean():.2f} "
        f"std {cumulative.std():.2f} -> {OUT_DIR}/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="explicit checkpoint dir; defaults to latest under save_dir/name",
    )
    parser.add_argument("--n", type=int, default=N, help="number of designs in the sweep")
    parser.add_argument("--steps", type=int, default=T, help="rollout length (env steps)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.n, args.steps)
