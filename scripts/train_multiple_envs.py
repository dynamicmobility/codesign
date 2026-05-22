import moplayground as mop
import minimal_mjx as mm
import mujoco
import argparse
import jax.numpy as jnp
from moplayground.envs.dmcontrol.cheetah import MOCheetah
from pathlib import Path

def change_rear_leg_length(cheetah_env:MOCheetah, length=1):
    """Modify the length of the 'bthigh' body in the cheetah environment."""

    # length_vec = mj_model.geom('bthigh').pos
    shin_pos = jnp.array([0.2, 0, -0.26])
    new_shin_pos = shin_pos*length
    midpoint = new_shin_pos/2

    # Construct a new cheetah env with new_shin_pos updated

    

    # return new_cheetah_env

parser = argparse.ArgumentParser()
parser.add_argument("length", type=str, default=0 ,help="Rear Leg Length")
args = parser.parse_args()
TRAIN_KWARGS = {}
EVAL_KWARGS  = {}

CONFIG_PATH = 'envs/codesign_cheetah.yaml'
train_config = mop.utils.read_config(CONFIG_PATH)
eval_config  = mop.utils.read_config(CONFIG_PATH)
    
print('Training', CONFIG_PATH)
env, env_cfg = mop.envs.create_environment(train_config, for_training=False, **TRAIN_KWARGS)
eval_env, _  = mop.envs.create_environment(eval_config, for_training=False, **EVAL_KWARGS)
name = train_config['save_dir'] + '/' + train_config['name']
# run = mm.utils.logging.initialize_wandb(
#     name    = name.replace('/', ''),
#     entity  = 'vmadabushi3-georgia-institute-of-technology',
#     project = 'codesign'
# )

# Modify the rear leg length
env = change_rear_leg_length(env, float(args.length))
# eval_env = change_rear_leg_length(eval_env, float(args.length))

# Dummy inference function:

def do_nothing(obs, rng):
    return jnp.zeros(env.action_size), 0.0

frames, reward_plotter, _, _ = mm.eval.rollout_policy(
        inference_fn    = do_nothing,
        env             = env,
        T               = 1,
        width           = 640,
        height          = 480,
        camera          = 'track',
        show_frames     = False
    )

mm.utils.plotting.save_video(
    frames,
    env.dt,
    Path(f'output/videos/{train_config['env']}-rollout.mp4')
)



# mop.learning.train_policy(train_config, env, eval_env, run)