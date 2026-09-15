import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
from itertools import combinations
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from moplayground.utils.pareto import get_nondominated
from codesign.eval.parallel_eval import rollout_so_parallel, build_grid_rollout_fn, grid_keys
from codesign.learning.inference import load_mo_design_hypernetwork
from codesign.utils.grid import Grid
import minimal_mjx as mm
import pandas as pd
import glob
from dynamo_figures import CompositeMode, CompositeImage
matplotlib.use("TKagg")

RUN_ID = "zpzu9ms5"
CONFIG_PATH = f"results/wandb-downloads/{RUN_ID}/config.yaml"
config     = mm.utils.config.create_config_dict(mop.utils.read_config(CONFIG_PATH))
# env        = codesign.cheetah(env_params=env_params, backend="jnp")
env, env_params = codesign.load_env(config=config, backend="np")
lower_bounds = np.array([env_params.codesign.low])
upper_bounds = np.array([env_params.codesign.high])
ROLLOUT_STEPS = 200

# This is obtained by running scripts/optimize/turbo_on_hypervolume.py
GENERALIST_DESIGN = [1.02006672, 1.71947823, 1.94088529, 1.47456715, 0.90478917, 1.47590946]
# These are obtained from plot_pareto_3d
RUN_DESIGN = [0.6152434, 0.8852895, 1.4163877, 1.1994246, 1.0384812, 0.52137005]
ENERGY_DESIGN = [1.293534, 0.66744065, 0.66819084, 0.55856186, 1.9247661, 1.0071616 ]
HEIGHT_DESIGN = [1.3124598, 1.9924996, 1.8101603, 1.8071598, 1.522273, 0.7453903]

OBJECTIVE_LABELS = ["Run", "Energy", "Height"]

GEN_VIDEO = False


inference_fn, params = load_mo_design_hypernetwork(config)

rollout = build_grid_rollout_fn(env, ROLLOUT_STEPS, inference_fn, deterministic=True)
tradeoffs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
designs = np.vstack((GENERALIST_DESIGN, RUN_DESIGN, ENERGY_DESIGN, HEIGHT_DESIGN))

tradeoff_settings = [([1, 0, 0], {"mode": CompositeMode.MIN_VALUE, "right_crop": [0, 640], "t_end": 0.9, "camera": "side_fixed_3"}),
                     ([0, 1, 0], {"mode": CompositeMode.MIN_VALUE, "right_crop": [0, 320], "t_end": 0.9, "camera": "side_fixed_3"}),
                     ([0, 0, 1], {"mode": CompositeMode.MIN_VALUE, "right_crop": [0, 400], "t_end": 0.9, "camera": "side_fixed_4"}),
                     ([1, 1, 1], {"mode": CompositeMode.MIN_VALUE, "right_crop": [20, 580], "t_end": 1.5, "camera": "side_fixed_3"})]

design_settings =   [(GENERALIST_DESIGN, {"top_crop": [0, 370], "color": "#000000"}),
                     (RUN_DESIGN,        {"top_crop": [0, 370], "color": "#74c5e4ff"}),
                     (ENERGY_DESIGN,     {"top_crop": [0, 370], "color": "#6fcfc7ff"}),
                     (HEIGHT_DESIGN,     {"top_crop": [0, 370], "color": "#fdafb2ff"})]

grid = Grid.crossed(designs, tradeoffs)

images_grid = []

for i in range(designs.shape[0]):
    design, d_settings = design_settings[i]
    row = []
    for j in range(tradeoffs.shape[0]):
        tradeoff, t_settings = tradeoff_settings[j]
        filename = f'scripts/icra/outputs/{RUN_ID}/collage/design{i}_tradeoff{j}.mp4'
        if(GEN_VIDEO):
            codesign.save_policy_rollout_video(
                config,
                out_path = filename,
                design=designs[i, :],
                tradeoff=tradeoff,
                n_steps=200,
                camera=t_settings["camera"],
                use_caption=False,
                width=640,
                height=480,
                robot_color=matplotlib.colors.to_rgba(d_settings["color"]),
            )
        merger = CompositeImage(
            mode=t_settings["mode"],
            video_path=filename,
            start_t=0.0,
            end_t=t_settings["t_end"],
            skip_frame=10,
            alpha=0.15
        )
        # Generate the composite
        top = d_settings["top_crop"][0]
        bottom = d_settings["top_crop"][1]
        left = t_settings["right_crop"][0]
        right = t_settings["right_crop"][1]
        result = merger.merge_images()
        result = result[top:bottom, left:right]
        row.append(result)
    row_concat = np.concatenate(row, axis=1)
    images_grid.append(row_concat)

image = np.concatenate(images_grid, axis=0)[:, :, [2, 1, 0]]
plt.imshow(image)
plt.imsave(f"scripts/icra/outputs/{RUN_ID}/collage/collage.png", image)
plt.axis('off')
plt.show()
