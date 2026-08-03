"""Single-instance (non-parallel) rollout of a policy on one Env, rendered to video.
"""

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src.mjx_env import render_array
from minimal_mjx.utils import plotting
from tqdm import tqdm

from codesign.envs import CodesignBase, MOCodesignBase
from minimal_mjx.eval import policy as policy_lib
from codesign.learning.inference import (
    load_design_hypernetwork,
    load_mo_design_hypernetwork,
)
from codesign.utils.model import normalize_design

from minimal_mjx.learning.inference import get_step_reset


def rollout_single_video(
    env: CodesignBase | MOCodesignBase,
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
    scene_option = plotting.get_mj_scene_option(contacts=False, com=False)
):
    """Roll a single Codesign env (one design) forward under ``policy`` and render it.

    Args:
        env: a CodesignBase env
        design: the single design to build the model via env.generate_model
        policy: ``policy(obs, key, t) -> (action, extras)``
        n_steps: maximum rollout length in env steps.
        seed: PRNG seed for reset + the action key stream.
        camera, width, height: rendering options (``width``/``height`` inferred from the
            model when ``None``).
        gen_video: whether to generate video
        show_progress: show a tqdm progress bar.

    Returns:
        ``(frames, traj)``: ``frames`` is a list of RGB arrays (or ``None``) and 
        ``traj`` the list of per-step env states.
    """
    mj_model = env.generate_model(design)

    if env.backend == 'jnp':
        model = mjx.put_model(mj_model)
    else:
        model = mj_model
    width, height = plotting.infer_frame_dim(model, width, height)

    step, reset = get_step_reset(env)

    rng = jax.random.PRNGKey(seed)
    state = reset(rng, model)
    traj = [state]
    
    # Setup reward plotting
    reward_plotter = plotting.RewardPlotter(state.metrics)
    data_plotter = plotting.MujocoPlotter()
    info_plotter = plotting.InfoPlotter(plotkey=None)
    data_plotter.add_row(state.data)
    
    for _ in tqdm(range(n_steps), disable=not show_progress):
        rng, sub = jax.random.split(rng)
        action, _ = policy(state.obs, sub, state.data.time)
        state = step(state, action, model)
        data_plotter.add_row(state.data)
        reward_plotter.add_row(state.metrics, state.reward)
        info_plotter.add_row(state.data.time, state.info)
        traj.append(state)
        if bool(state.done):
            break

    frames = None
    if gen_video:
        print("Generating video...")
        frames = render_array(
            mj_model, traj, height, width, camera, scene_option=scene_option
        )
    return frames, traj, reward_plotter, data_plotter, info_plotter


def rollout_design_hypernetwork_video(
    env,
    config: dict,
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
    eval_design = None,
):
    """Render a trained design-hypernetwork policy on a single design.

    Loads the checkpoint, builds the (unbatched) design-conditioned policy for ``design``,
    and rolls it out via :func:`rollout_single_video`. Single-instance analogue of
    :func:`codesign.eval.parallel_eval.rollout_design_hypernetwork`.

    Returns ``(frames, traj)`` (see :func:`rollout_single_video`).
    """
    if(eval_design is None):
        eval_design = design
    design       = np.asarray(design, np.float32).reshape(-1) # flatten
    design_input = normalize_design(jnp.asarray(design), config=config)

    # Trained, design-conditioned policy for this single design (1-D design -> unbatched).
    inference_fn, params = load_design_hypernetwork(config, path=checkpoint_path)
    base_policy          = inference_fn(params, design_input, deterministic=deterministic)
    policy               = policy_lib.from_inference_fn(base_policy)

    return rollout_single_video(
        env, eval_design, policy, n_steps,
        seed=seed, camera=camera, width=width, height=height, gen_video=gen_video,
    )


def rollout_mo_design_hypernetwork_video(
    env,
    config,
    design,
    tradeoff,
    n_steps: int,
    *,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    camera: str | None = None,
    width: int | None = None,
    height: int | None = None,
    gen_video: bool = True,
    eval_design = None,
):
    """Render a trained MO design-hypernetwork policy ``H(d, w)`` on one ``(design, w)``.

    Returns ``(frames, traj, reward_plotter, data_plotter, info_plotter)`` via :func:`rollout_single_video`.
    """
    if(eval_design is None):
        eval_design = design
    design       = np.asarray(design, np.float32).reshape(-1) # flatten
    design_input = normalize_design(jnp.asarray(design), config=config)

    tradeoff_arr   = np.asarray(tradeoff, np.float32).reshape(-1)  # (num_objectives,)
    tradeoff_arr   = tradeoff_arr / np.sum(tradeoff_arr)  # normalize onto the simplex
    tradeoff_input = jnp.asarray(tradeoff_arr)

    # Trained, (design, tradeoff)-conditioned policy (1-D inputs -> unbatched).
    inference_fn, params = load_mo_design_hypernetwork(config, path=checkpoint_path)
    base_policy = inference_fn(
        params, design_input, tradeoff_input, deterministic=deterministic
    )
    policy = policy_lib.from_inference_fn(base_policy)

    return rollout_single_video(
        env, eval_design, policy, n_steps,
        seed=seed, camera=camera, width=width, height=height, gen_video=gen_video,
    )
