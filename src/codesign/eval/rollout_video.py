"""Single-instance (non-parallel) rollout of a policy on one Env, rendered to video.
"""

from dataclasses import dataclass
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
    load_mo_design_predictor_hypernetwork,
)
from codesign.utils.model import normalize_design, unnormalize_design
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
    d = np.asarray(design, np.float32).reshape(-1)
    mj_model = env.generate_model(d)

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


def _simplex(tradeoff, num_objectives):
    """CLI tradeoff values -> a ``(num_objectives,)`` array summing to one."""
    if tradeoff is None:
        return np.full((num_objectives,), 1.0 / num_objectives, np.float32)
    w = np.asarray(tradeoff, np.float32).reshape(-1)
    return w / np.sum(w)


def rollout_mo_design_predictor_hypernetwork_video(
    env,
    config,
    tradeoff,
    n_steps: int,
    *,
    design_tradeoff=None,
    eval_design=None,
    sample_design: bool = False,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    camera: str | None = None,
    width: int | None = None,
    height: int | None = None,
    gen_video: bool = True,
):
    """Render ``H(d, w)`` on a design the predictor picks: ``d ~ f(. | w')``.

    ``design_tradeoff`` is the ``w'`` fed to the predictor and defaults to ``tradeoff``,
    the ``w`` conditioning the policy. Decoupling them tests the predictor's alignment
    (sweep ``w'`` on a fixed ``w``) and its specificity (sweep ``w`` on a fixed ``w'``).

    Returns ``(frames, traj, reward_plotter, data_plotter, info_plotter, design)`` where
    ``design`` is the predicted design in physical units.
    """
    num_objectives = len(config["env_config"]["reward"]["optimization"]["objectives"])
    tradeoff_input = jnp.asarray(_simplex(tradeoff, num_objectives))
    design_tradeoff_input = jnp.asarray(
        _simplex(tradeoff if design_tradeoff is None else design_tradeoff, num_objectives)
    )

    inference_fn, design_predictor_inference_fn, params = (
        load_mo_design_predictor_hypernetwork(config, path=checkpoint_path)
    )
    # The predictor emits designs already normalized to [0, 1] (1-D input -> unbatched).
    design_input, _ = design_predictor_inference_fn(
        params[2],
        design_tradeoff_input,
        deterministic=not sample_design,
        key_sample=jax.random.PRNGKey(seed),
    )
    design = np.asarray(unnormalize_design(design_input, config=config)).reshape(-1)
    if eval_design is None:
        eval_design = design

    base_policy = inference_fn(
        params, design_input, tradeoff_input, deterministic=deterministic
    )
    policy = mm.from_inference_fn(base_policy)

    return (
        *rollout_single_video(
            env, eval_design, policy, n_steps,
            seed=seed, camera=camera, width=width, height=height, gen_video=gen_video,
        ),
        design,
    )


def default_video_design(config):
    """Design to roll out when the caller doesn't name one.
    """
    if config["algorithm"] == "ppo":
        return np.asarray(config['env_config']['codesign']["default_design"], np.float32).reshape(-1)

    design = config["env_config"]['codesign']
    low  = np.asarray(design["low"], np.float32).reshape(-1)
    high = np.asarray(design["high"], np.float32).reshape(-1)
    return 0.5 * (low + high)  # (design_dim,) box midpoint


def _writable_frame(frame):
    """A frame OpenCV can draw into, copying only when it has to.
    """
    frame = np.asarray(frame)
    if frame.flags.writeable and frame.flags.c_contiguous:
        return frame
    return np.array(frame, order="C")


def _captioned_frame(frame, caption):
    """``frame`` with ``caption`` drawn into its top-left corner.
    """
    return mm.add_text_to_frame(
        _writable_frame(frame),
        caption,
        org               = (10, 28),
        size              = 0.6,      # 640px-wide frames; larger overflows the caption
        thickness         = 2,        # 1 gets swallowed by the outline's antialiasing
        color             = (255, 255, 255),
        outline_color     = (0, 0, 0),
        outline_thickness = 2,
    )


def extreme_tradeoffs_with_labels(config):
    """The corners of the objective simplex, as ``[(label, w), ...]``.
    """
    labels  = objective_labels(config["env_config"]["reward"]["optimization"]["objectives"])
    corners = np.eye(len(labels), dtype=np.float32)
    return list(zip(labels, corners))


@dataclass
class RolloutVideo:
    """A rendered rollout plus everything needed to caption, save and score it.

    Attributes:
        frames: list of ``(height, width, 3)`` uint8 RGB arrays.
        traj: list of per-step env states.
        reward_plotter, data_plotter, info_plotter: the rollout's recorded traces.
        design: ``(design_dim,)`` design the policy was conditioned on (for the design
            predictor, the design it predicted).
        eval_design: ``(design_dim,)`` design the mujoco model was built from.
        tradeoff: ``(num_objectives,)`` scalarization ``w`` conditioning the policy, or
            ``None`` for the single-objective algorithms.
        design_tradeoff: ``(num_objectives,)`` scalarization ``w'`` fed to the design
            predictor, or ``None`` when it is not decoupled from ``w``.
        caption: one-line description of the rollout.
        dt: env timestep [s], i.e. the video's frame period.
        path: where the video was written, once it has been.
    """
    frames          : list
    traj            : list
    reward_plotter  : mm.RewardPlotter
    data_plotter    : mm.MujocoPlotter
    info_plotter    : mm.InfoPlotter
    design          : np.ndarray
    eval_design     : np.ndarray
    tradeoff        : np.ndarray | None
    design_tradeoff : np.ndarray | None
    caption         : str
    dt              : float
    path            : Path | None = None


def _num_objectives(config):
    """How many objectives the env's reward is split into."""
    return len(config["env_config"]["reward"]["optimization"]["objectives"])


def _default_designs(config, design, eval_design):
    """``(design, eval_design)``, filling ``None`` with the config default / ``design``."""
    design = default_video_design(config) if design is None else design
    design = np.asarray(design, np.float32).reshape(-1)
    if eval_design is None:
        return design, design
    return design, np.asarray(eval_design, np.float32).reshape(-1)


def rollout_caption(config, design, eval_design=None, tradeoff=None, design_tradeoff=None):
    """One-line description of a rollout, for video overlays and plot titles.

    ``eval_design`` and ``design_tradeoff`` are only named when they differ from
    ``design`` and ``tradeoff``, i.e. when the rollout is deliberately mismatched.
    """
    fmt = lambda x: np.round(np.asarray(x, np.float32).reshape(-1), 3)

    caption = f"{config['env']} d={fmt(design)}"
    if eval_design is not None and not np.array_equal(fmt(eval_design), fmt(design)):
        caption += f" eval d={fmt(eval_design)}"
    if tradeoff is not None:
        caption += f" w={fmt(tradeoff)}"
    if design_tradeoff is not None and not np.array_equal(fmt(design_tradeoff), fmt(tradeoff)):
        caption += f" w'={fmt(design_tradeoff)}"
    return caption


def _rollout_ppo_video(
    env, config, eval_design, n_steps, *, checkpoint_path=None, seed=0,
    camera=None, width=None, height=None,
):
    """Roll the fixed-design PPO policy out on the scalarized (single-objective) env.

    The policy is unconditioned, so ``eval_design`` only selects the model it runs on.
    ``load_env`` already wraps ppo envs; the wrap here is for callers that pass a raw one.
    """
    base_policy = mm.load_policy(config, deterministic=True, checkpoint_path=checkpoint_path)
    policy      = mm.from_inference_fn(base_policy)
    so_env      = (
        env if isinstance(env, CodesignMO2SO)
        else CodesignMO2SO(
            env, config["env_config"]["reward"]["optimization"]["default_scalarization"]
        )
    )
    return rollout_single_video(
        so_env, eval_design, policy, n_steps,
        seed=seed, camera=camera, width=width, height=height,
    )


def rollout_policy_video(
    config,
    *,
    env=None,
    design=None,
    eval_design=None,
    tradeoff=None,
    design_tradeoff=None,
    sample_design: bool = False,
    n_steps: int = 500,
    checkpoint_path: str | None = None,
    seed: int = 0,
    camera: str | None = None,
    width: int | None = 640,
    height: int | None = 480,
) -> RolloutVideo:
    """Roll the policy trained by ``config`` out on one design and render it.

    Dispatches on ``config['algorithm']`` to the matching ``rollout_*_video`` above, so
    callers need not know which conditioning inputs their checkpoint takes.

    Args:
        config: the run config (as saved to ``config.yaml`` at train time).
        env: (optional) a renderable (``backend='np'``) env; loaded from ``config`` if
            omitted. Pass one in to reuse it across several videos.
        design: design fed to the hypernetwork; defaults to :func:`default_video_design`.
            Ignored by ``mo_design_predictor_hypernetwork``, which predicts its own.
        eval_design: design the model is *built* from; defaults to ``design`` (to the
            predicted design for ``mo_design_predictor_hypernetwork``).
        tradeoff: objective scalarization ``w`` conditioning the policy (MO algorithms
            only); defaults to uniform over the objectives.
        design_tradeoff: scalarization ``w'`` fed to the design predictor; defaults to
            ``tradeoff``. Decoupling the two tests the predictor's alignment.
        sample_design: sample from ``f(d | w')`` instead of taking its mode.
        n_steps: rollout length in env steps.
        checkpoint_path: explicit checkpoint dir; defaults to the latest under
            ``save_dir/name``.
        seed, camera, width, height: rollout/rendering options.

    Returns:
        A :class:`RolloutVideo` (its ``path`` unset -- see :func:`write_rollout_video`).
    """
    algorithm = config["algorithm"]
    if env is None:
        env, _ = load_env(config, backend="np")  # renderable (mujoco, host-side) env

    if algorithm == "ppo":
        design, eval_design       = _default_designs(config, design, eval_design)
        tradeoff, design_tradeoff = None, None
        rollout = _rollout_ppo_video(
            env, config, eval_design, n_steps, checkpoint_path=checkpoint_path,
            seed=seed, camera=camera, width=width, height=height,
        )

    elif algorithm == "design_hypernetwork":
        design, eval_design       = _default_designs(config, design, eval_design)
        tradeoff, design_tradeoff = None, None
        rollout = rollout_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, n_steps=n_steps,
            checkpoint_path=checkpoint_path, seed=seed,
            camera=camera, width=width, height=height,
        )

    elif algorithm == "mo_design_hypernetwork":
        design, eval_design       = _default_designs(config, design, eval_design)
        tradeoff, design_tradeoff = _simplex(tradeoff, _num_objectives(config)), None
        rollout = rollout_mo_design_hypernetwork_video(
            env, config, design=design, eval_design=eval_design, tradeoff=tradeoff,
            n_steps=n_steps, checkpoint_path=checkpoint_path, seed=seed,
            camera=camera, width=width, height=height,
        )

    elif algorithm == "mo_design_predictor_hypernetwork":
        # d ~ f(. | w'), so `design` is unused; `eval_design` still overrides the model.
        tradeoff = _simplex(tradeoff, _num_objectives(config))
        if design_tradeoff is not None:
            design_tradeoff = _simplex(design_tradeoff, _num_objectives(config))
        *rollout, design = rollout_mo_design_predictor_hypernetwork_video(
            env, config, tradeoff=tradeoff, design_tradeoff=design_tradeoff,
            eval_design=eval_design, sample_design=sample_design, n_steps=n_steps,
            checkpoint_path=checkpoint_path, seed=seed,
            camera=camera, width=width, height=height,
        )
        if eval_design is None:
            eval_design = design
        else:
            eval_design = np.asarray(eval_design, np.float32).reshape(-1)

    else:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; expected 'ppo', 'design_hypernetwork', "
            "'mo_design_hypernetwork' or 'mo_design_predictor_hypernetwork'."
        )

    frames, traj, reward_plotter, data_plotter, info_plotter = rollout
    return RolloutVideo(
        frames          = frames,
        traj            = traj,
        reward_plotter  = reward_plotter,
        data_plotter    = data_plotter,
        info_plotter    = info_plotter,
        design          = design,
        eval_design     = eval_design,
        tradeoff        = tradeoff,
        design_tradeoff = design_tradeoff,
        caption         = rollout_caption(config, design, eval_design, tradeoff, design_tradeoff),
        dt              = env.dt,
    )


def write_rollout_video(
    rollout: RolloutVideo,
    out_path,
    *,
    run: wandb.Run | None = None,
    log_key: str = "rollout",
) -> RolloutVideo:
    """Caption ``rollout``'s frames, write them to ``out_path``, optionally log to W&B.

    Returns the same ``rollout``, with ``path`` set to where the video was written.
    """
    frames   = [_captioned_frame(frame, rollout.caption) for frame in rollout.frames]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # save_video otherwise prompts
    mm.save_video(frames, rollout.dt, out_path)

    if run is not None:
        run.log(
            {log_key: wandb.Video(str(out_path), caption=rollout.caption, format="mp4")}
        )

    rollout.path = out_path
    return rollout


def save_policy_rollout_video(
    config,
    out_path,
    *,
    env=None,
    design=None,
    eval_design=None,
    tradeoff=None,
    design_tradeoff=None,
    sample_design: bool = False,
    n_steps: int = 500,
    checkpoint_path: str | None = None,
    seed: int = 0,
    camera: str | None = None,
    width: int | None = 640,
    height: int | None = 480,
    run: wandb.Run | None = None,
    log_key: str = "rollout",
) -> RolloutVideo:
    """Roll out the trained policy for ``config`` and write the video to ``out_path``.

    :func:`rollout_policy_video` followed by :func:`write_rollout_video`; see those for
    the arguments. ``run``/``log_key`` are the (optional) W&B run to log the video to.

    Returns:
        The :class:`RolloutVideo`, whose ``path`` is where the video was written.
    """
    rollout = rollout_policy_video(
        config, env=env, design=design, eval_design=eval_design, tradeoff=tradeoff,
        design_tradeoff=design_tradeoff, sample_design=sample_design, n_steps=n_steps,
        checkpoint_path=checkpoint_path, seed=seed,
        camera=camera, width=width, height=height,
    )
    print(f"rendered {len(rollout.traj)} steps: {rollout.caption}")
    return write_rollout_video(rollout, out_path, run=run, log_key=log_key)
