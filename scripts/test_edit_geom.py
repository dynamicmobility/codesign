# Throw a ball at 100 different velocities.

import jax
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import mujoco.viewer
import time
import numpy as np
import jax.numpy as jnp


# Compile the model
spec = mujoco.MjSpec.from_file("envs/cheetah.xml")
m = spec.compile()

length = 1.5
shin_pos = jnp.array([0.2, 0, -0.26])
new_shin_pos = shin_pos*length
midpoint = new_shin_pos/2
spec.body("bshin").pos = new_shin_pos
thigh_geom = spec.geom("bthigh")
thigh_geom.pos = midpoint  # Change to desired position (x, y, z)
thigh_geom.size[1] = jnp.linalg.norm(new_shin_pos)/2
spec.body("torso").pos = [0, 0, 1]

m = spec.compile()
d = mujoco.MjData(m)

# redo put

with mujoco.viewer.launch_passive(m, d) as viewer:

  # Enable wireframe rendering of the entire scene.
#   viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
#   viewer.sync()

  while viewer.is_running():
    step_start = time.time()
    # Step the physics.
    mujoco.mj_step(m, d)
    viewer.sync()
    this_step_time = time.time() - step_start
    if(this_step_time < 0.01):
        time.sleep(0.01 - this_step_time)

# vel = jax.numpy.arange(0.0, 1.0, 0.01)
# pos = jax.jit(batched_step)(vel)
# print(pos)

# mjx_model = mjx.put_model(m)

# @jax.vmap
# def batched_step(vel):
#   mjx_data = mjx.make_data(mjx_model)
#   qvel = mjx_data.qvel.at[0].set(vel)
#   mjx_data = mjx_data.replace(qvel=qvel)
#   pos = mjx.step(mjx_model, mjx_data).qpos
#   return pos

# vel = jax.numpy.arange(0.0, 1.0, 0.01)
# pos = jax.jit(batched_step)(vel)
# print(pos)