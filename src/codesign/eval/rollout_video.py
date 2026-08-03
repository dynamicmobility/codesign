"""Single-instance (non-parallel) rollout of a policy on one Env, rendered to video.
"""

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import wandb
from mujoco import mjx
from mujoco_playground._src.mjx_env import render_array
from tqdm import tqdm

from codesign.envs import CodesignBase, MOCodesignBase, CodesignMO2SO, load_env
from minimal_mjx.eval import policy as policy_lib
from codesign.learning.inference import (
    load_design_hypernetwork,
    load_mo_design_hypernetwork,
)
from codesign.utils.model import normalize_design
from codesign.utils.plotting import objective_labels

# from minimal_mjx.learning.inference import get_step_reset, load_policy
import minimal_mjx as mm


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
    scene_option = mm.get_mj_scene_option(contacts=False, com=False)
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
    width, height = mm.infer_frame_dim(model, width, height)

    step, reset = mm.get_step_reset(env)

    rng = jax.random.PRNGKey(seed)
    state = reset(rng, model)
    traj = [state]
    
    # Setup reward plotting
    reward_plotter = mm.RewardPlotter(state.metrics)
    data_plotter = mm.MujocoPlotter()
    info_plotter = mm.InfoPlotter(plotkey=None)
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
    policy               = mm.from_inference_fn(base_policy)

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
    policy = mm.from_inference_fn(base_policy)

    return rollout_single_video(
        env, eval_design, policy, n_steps,
        seed=seed, camera=camera, width=width, height=height, gen_video=gen_video,
    )


def default_video_design(config):
    """Design to roll out when the caller doesn't name one.
    """
    learning_params = config["learning_params"]
    if config["algorithm"] == "ppo":
        return np.asarray(learning_params["default_design"], np.float32).reshape(-1)

    design_params = learning_params["design_params"]
    low  = float(design_params["design_low"])
    high = float(design_params["design_high"])
    dim  = int(design_params["design_dim"])
    return np.full((dim,), 0.5 * (low + high), np.float32)


def _writable_frame(frame):
    """A frame OpenCV can draw into, copying only when it has to.
    """
    frame = np.asarray(frame)
    if frame.flags.writeable and frame.flags.c_contiguous:
        return frame
    return np.array(frame, order="C")


def extreme_tradeoffs_with_labels(config):
    """The corners of the objective simplex, as ``[(label, w), ...]``.
    """
    labels  = objective_labels(config["env_config"]["reward"]["optimization"]["objectives"])
    corners = np.eye(len(labels), dtype=np.float32)
    return list(zip(labels, corners))


def save_policy_rollout_video(
    config,
    out_path,
    *,
    env=None,
    design=None,
    eval_design=None,
    tradeoff=None,
    n_steps: int = 500,
    checkpoint_path: str | None = None,
    seed: int = 0,
    camera: str | None = None,
    width: int | None = 640,
    height: int | None = 480,
    run: wandb.Run | None = None,
    log_key: str = "rollout",
) -> Path:
    """Roll out the trained policy for ``config`` and write the video to ``out_path``.

    Args:
        config: the run config (as saved to ``config.yaml`` at train time).
        out_path: where to write the ``.mp4``; parent dirs are created.
        env: (optional) a renderable (``backend='np'``) env; loaded from ``config`` if
            omitted. Pass one in to reuse it across several videos.
        design: design fed to the hypernetwork; defaults to :func:`default_video_design`.
        eval_design: design the model is *built* from; defaults to ``design``.
        tradeoff: objective scalarization ``w`` (``mo_design_hypernetwork`` only);
            defaults to uniform over the objectives.
        n_steps: rollout length in env steps.
        checkpoint_path: explicit checkpoint dir; defaults to the latest under
            ``save_dir/name``.
        seed, camera, width, height: rollout/rendering options.
        run: (optional) W&B run; the video is logged to it under ``log_key``.
        log_key: W&B key to log the video under.

    Returns:
        The ``Path`` the video was written to.
    """
    algorithm = config["algorithm"]
    if env is None:
        env, _ = load_env(config, backend="np")  # renderable (mujoco, host-side) env
    if design is None:
        design = default_video_design(config)
    if eval_design is None:
        eval_design = design

    caption = f"{config['env']} d={np.asarray(design).reshape(-1)}"

    if algorithm == "mo_design_hypernetwork":
        if tradeoff is None:
            num_objectives = len(
                config["env_config"]["reward"]["optimization"]["objectives"]
            )
            tradeoff = [1.0 / num_objectives] * num_objectives
        caption = f"{caption} w={np.round(np.asarray(tradeoff), 3)}"
        frames, traj, _, _, _ = rollout_mo_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, tradeoff=tradeoff,
            n_steps=n_steps, checkpoint_path=checkpoint_path, seed=seed,
            camera=camera, width=width, height=height,
        )

    elif algorithm == "design_hypernetwork":
        frames, traj, _, _, _ = rollout_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, n_steps=n_steps,
            checkpoint_path=checkpoint_path, seed=seed,
            camera=camera, width=width, height=height,
        )

    elif algorithm == "ppo":
        # Single fixed design -> a plain (unconditioned) policy over the scalarized env.
        base_policy = mm.load_policy(
            config, deterministic=True, checkpoint_path=checkpoint_path
        )
        policy = mm.from_inference_fn(base_policy)
        so_env = (
            env if isinstance(env, CodesignMO2SO)
            else CodesignMO2SO(env, config["learning_params"]["reward_objective_weights"])
        )
        frames, traj, _, _, _ = rollout_single_video(
            so_env, eval_design, policy, n_steps,
            seed=seed, camera=camera, width=width, height=height,
        )

    else:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; expected 'ppo', 'design_hypernetwork' "
            "or 'mo_design_hypernetwork'."
        )

    frames = [
        mm.add_text_to_frame(
            _writable_frame(frame),
            caption,
            org               = (10, 28),
            size              = 0.6,      # 640px-wide frames; larger overflows the caption
            thickness         = 2,        # 1 gets swallowed by the outline's antialiasing
            color             = (255, 255, 255),
            outline_color     = (0, 0, 0),
            outline_thickness = 2,
        )
        for frame in frames
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # save_video otherwise prompts
    mm.save_video(frames, env.dt, out_path)
    print(f"rendered {len(traj)} steps for design d={np.asarray(design).reshape(-1)}.")

    if run is not None:
        run.log({log_key: wandb.Video(str(out_path), caption=caption, format="mp4")})

    return out_path
