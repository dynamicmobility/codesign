import minimal_mjx as mm
from minimal_mjx.utils import plotting
from mujoco_playground._src.mjx_env import MjxEnv
import jax
import numpy as np
from tqdm import tqdm

def make_dummy_inference_fn(env: mm.envs.SwappableBase, mode='zero'):
    """Makes a 'fake' inference function for simulating the environment without 
    a policy. Modes are 'zero' (which sends all zeros as the action) and 
    'random' (which sends random actions in [-1, 1])"""
    inference_fn = None
    match mode:
        case 'zero':
            inference_fn = lambda obs, rng: (np.zeros(env.action_size), None)
        case 'random':
            inference_fn = lambda obs, rng, size=env.action_size: (2 * np.random.random(size) - 1, None)
        case _:
            raise Exception(f'Unknown inference type {mode}')

    return inference_fn

def rollout_policy(
    inference_fn,
    env           : MjxEnv,
    T             = 10.0,
    info_init_fn  = lambda state: state.info,
    info_step_fn  = lambda state: state.info,
    info_plot_key = None,
    width         = None,
    height        = None,
    gen_vid       = True,
    show_progress = True,
    camera        = None,
) -> tuple[list, plotting.RewardPlotter, plotting.MujocoPlotter, plotting.InfoPlotter]:
    width, height = plotting.infer_frame_dim(env.mj_model, width, height)
    step, reset = mm.learning.inference.get_step_reset(env)

    # Set up the environment
    rng = jax.random.PRNGKey(np.random.randint(0, 100000))
    state: mm.EnvState = reset(rng)
    
    # Initialize the state
    initial_info = state.info | info_init_fn(state)
    state = state.replace(info=initial_info)

    env.set_mj_model(state.info['model'])
    # Setup reward plotting
    reward_plotter = plotting.RewardPlotter(state.metrics)
    data_plotter = plotting.MujocoPlotter()
    info_plotter = plotting.InfoPlotter(plotkey=info_plot_key)
    data_plotter.add_row(state.data)

    # Rollout and record data
    N = int(T / env.dt)
    traj = [state]
    for i in tqdm(range(N), disable=not show_progress):
        ctrl, _ = inference_fn(state.obs, rng)
        state = step(state, ctrl)
        state = state.replace(info=state.info | info_step_fn(state))
        data_plotter.add_row(state.data)
        reward_plotter.add_row(state.metrics, state.reward)
        info_plotter.add_row(state.data.time, state.info)
        traj.append(state)
        
        if state.done:
            break

    scene_option = plotting.get_mj_scene_option(contacts=False, com=False)

    if gen_vid:
        print('Generating video...')
        frames = env.render(
            trajectory   = traj,
            camera       = camera,
            height       = height,
            width        = width,
            scene_option = scene_option,
        )
    else:
        frames = None
    return frames, reward_plotter, data_plotter, info_plotter