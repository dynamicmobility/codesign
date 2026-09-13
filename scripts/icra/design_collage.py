import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from itertools import combinations
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated
import minimal_mjx as mm
import pandas as pd
import glob
matplotlib.use("TKagg")

CHEETAH_CONFIG = "config/mo_design_hypernetwork/cheetah6D.yaml"
WALKER_CONFIG = "config/mo_design_hypernetwork/walker6D.yaml"
TILE_WIDTH = 250


train_config = mm.read_config(CHEETAH_CONFIG)
cheetah_env, _ = codesign.load_env(train_config, backend='np')
policy_cheetah = mm.make_open_loop_policy("zero", cheetah_env.action_size)

train_config = mm.read_config(WALKER_CONFIG)
walker_env, _ = codesign.load_env(train_config, backend='np')
policy_walker = mm.make_open_loop_policy("zero", walker_env.action_size)

designs = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
cheetah_images = []
walker_images = []

for i in range(designs.shape[0]):
    frames, traj, reward_plotter, data_plotter, info_plotter = codesign.rollout_single_video(
        cheetah_env, designs[i, :], policy_cheetah, n_steps=0, camera='side_fixed_2', width=TILE_WIDTH, height=480,
        enable_termination=True,
    )
    cheetah_images.append(frames[0][200:, ...])

    frames, traj, reward_plotter, data_plotter, info_plotter = codesign.rollout_single_video(
        walker_env, designs[i, :], policy_walker, n_steps=0, camera='side_fixed_2', width=TILE_WIDTH, height=480,
        enable_termination=True,
    )
    walker_images.append(frames[-1])

cheetahs = np.concatenate(cheetah_images, axis=1)
walkers = np.concatenate(walker_images, axis=1)
image = np.concatenate((cheetahs, walkers), axis=0)[:700, ...]
plt.imshow(image)
plt.axis('off')
plt.imsave("scripts/icra/outputs/collage.png", image)
plt.show()
