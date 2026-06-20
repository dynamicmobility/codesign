"""Single-instance (non-parallel) rollout of a policy on one MAI design, rendered to video.

The single-env counterpart of :mod:`codesign.eval.parallel_eval`: instead of stacking a
sweep of designs and scanning them together, this builds *one* design-specific model,
steps a single env under a policy, and renders the trajectory to frames. The generic core
:func:`rollout_single` takes any ``policy(obs, key, t)`` (open-loop or trained — see
:mod:`codesign.eval.policies`); :func:`rollout_design_hypernetwork_video` is a thin wrapper
that loads a trained design-conditioned policy for the given design.
"""

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src.mjx_env import render_array
from minimal_mjx.utils import plotting
from tqdm import tqdm

from codesign.eval import policies as policy_lib
from codesign.learning.inference import load_design_hypernetwork
from codesign.utils.model import normalize_design


def rollout_single(
    env,
    design,
    policy,
    n_steps: int,
    *,
    seed: int = 0,
    camera: str | None = None,
    width: int | None = None,
    height: int | None = None,
    gen_video: bool = True,
    show_progress: bool = True,
):
    """Roll a single MAI env (one design) forward under ``policy`` and render it.

    Builds the design-specific model via ``env.generate_model(d)``, resets/steps one env
    under ``policy(obs, key, t)`` for up to ``n_steps`` (stopping early on termination),
    and renders the trajectory.

    Args:
        env: a model-as-input env (e.g. ``MAICheetah``) exposing ``generate_model(d)`` and
            ``reset(rng, model)`` / ``step(state, action, model)``.
        design: the single design ``d`` (scalar or ``(design_dim,)``); the model is built
            from its first entry.
        policy: ``policy(obs, key, t) -> (action, extras)`` producing a single
            (unbatched) action — see :mod:`codesign.eval.policies`.
        n_steps: maximum rollout length in env steps.
        seed: PRNG seed for reset + the action key stream.
        camera, width, height: rendering options (``width``/``height`` inferred from the
            model when ``None``).
        gen_video: render frames (``False`` returns ``None`` frames, e.g. for a dry run).
        show_progress: show a tqdm progress bar.

    Returns:
        ``(frames, traj)`` — ``frames`` is a list of RGB arrays (or ``None``) and ``traj``
        the list of per-step env states.
    """
    d = float(np.asarray(design).reshape(-1)[0])
    mj_model = env.generate_model(d)
    mjx_model = mjx.put_model(mj_model)
    width, height = plotting.infer_frame_dim(mj_model, width, height)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    state = reset(rng, mjx_model)
    traj = [state]
    t = 0.0
    for _ in tqdm(range(n_steps), disable=not show_progress):
        rng, sub = jax.random.split(rng)
        action, _ = policy(state.obs, sub, t)
        state = step(state, action, mjx_model)
        traj.append(state)
        t += env.dt
        if bool(state.done):
            break

    frames = None
    if gen_video:
        print("Generating video...")
        scene_option = plotting.get_mj_scene_option(contacts=False, com=False)
        frames = render_array(
            mj_model, traj, height, width, camera, scene_option=scene_option
        )
    return frames, traj


def rollout_design_hypernetwork_video(
    env,
    config,
    design,
    n_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    camera: str | None = None,
    width: int | None = None,
    height: int | None = None,
    gen_video: bool = True,
):
    """Render a trained design-hypernetwork policy on a single design.

    Loads the checkpoint, builds the (unbatched) design-conditioned policy for ``design``,
    and rolls it out via :func:`rollout_single`. Single-instance analogue of
    :func:`codesign.eval.parallel_eval.rollout_design_hypernetwork`.

    Returns ``(frames, traj)`` (see :func:`rollout_single`).
    """
    design_params = config["learning_params"]["design_params"]
    design_low = float(design_params["design_low"])
    design_high = float(design_params["design_high"])

    design_arr = np.asarray(design, np.float32).reshape(-1)  # (design_dim,)
    design_input = normalize_design(jnp.asarray(design_arr), design_low, design_high)

    # Trained, design-conditioned policy for this single design (1-D design -> unbatched).
    inference_fn, params = load_design_hypernetwork(config, path=checkpoint_path)
    base_policy = inference_fn(params, design_input, deterministic=deterministic)
    policy = policy_lib.from_inference_fn(base_policy)

    return rollout_single(
        env, design_arr, policy, n_steps,
        seed=seed, camera=camera, width=width, height=height, gen_video=gen_video,
    )
