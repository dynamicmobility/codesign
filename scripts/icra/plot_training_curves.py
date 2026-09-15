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
matplotlib.use("TKAgg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from scripts import icra

DOWNLOAD_FILES = False
BASE_FILEPATH = "scripts/icra/outputs"
HV_FILE = "Pareto-Optimal Design Hypervolume"
SP_FILE = "Pareto-Optimal Design Spacing"

def download_charts(entity, project, run_id, metrics, out_dir, x_key="_step"):
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        df = run.history(keys=[metric])
        if df.empty:
            continue

        name = metric.replace("/", "_")
        df.to_csv(out_dir / f"{name}.csv", index=False)

def plot_hv(ax, steps, data, **plot_kwargs):
    step_clip = steps[steps < icra.MO_CHECKPOINT]
    data_clip = data[steps < icra.MO_CHECKPOINT] 
    ax.plot(step_clip, data_clip, **plot_kwargs)



for i, config in enumerate(icra.FINAL_CONFIGS):
    robot_config = icra.FINAL_CONFIGS[config]

    for alg_config in robot_config:
        alg = robot_config[alg_config]
        metrics = ["Pareto-Optimal Design Hypervolume", "Pareto-Optimal Design Spacing"]
        if(type(alg) == icra.Run and alg.run_id is not None):
            filepath = f"{BASE_FILEPATH}/{alg.run_id}/"
            if(DOWNLOAD_FILES):
                download_charts(
                    entity  = 'vmadabushi3-georgia-institute-of-technology',
                    project = 'codesign',
                    run_id  = alg.run_id,
                    metrics = metrics,
                    out_dir = filepath,
                )
                print(f"Downloaded {alg.run_id}")

cheetah_mdh_hv = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["cheetah"]["MDH"].run_id}/{HV_FILE}.csv")
cheetah_mlp_hv = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["cheetah"]["MLP"].run_id}/{HV_FILE}.csv")

cheetah_mdh_sp = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["cheetah"]["MDH"].run_id}/{SP_FILE}.csv")
cheetah_mlp_sp = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["cheetah"]["MLP"].run_id}/{SP_FILE}.csv")

ax1 = plt.subplot(2,1,1)
plot_hv(ax1, cheetah_mdh_hv["_step"], cheetah_mdh_hv[HV_FILE], label = "Hypernetwork", color = "#236E7D", marker='.')
plot_hv(ax1, cheetah_mlp_hv["_step"], cheetah_mlp_hv[HV_FILE], label = "MLP", color = "#9136AA", marker='.')
# ax1.legend()
codesign.dress_axis(ax1)
ax1.set_ylabel("Hypervolume")
ax1.set_xlabel("")
ax1.tick_params(axis="x", labelbottom=False)
ax2 = ax1.twinx()
plot_hv(ax2, cheetah_mdh_sp["_step"], cheetah_mdh_sp[SP_FILE], label = "Hypernetwork", color = "#236E7D", linestyle = '--', marker='.')
plot_hv(ax2, cheetah_mlp_sp["_step"], cheetah_mlp_sp[SP_FILE], label = "MLP", color = "#9136AA", linestyle = '--', marker='.')
ax2.set_ylim([0, 0.3])
ax2.set_ylabel("Spacing")
ax2.set_xlabel("")
ax2.tick_params(axis="x", labelbottom=False)
ax2.set_title("Cheetah (top), Walker (bottom)")
codesign.dress_axis(ax2)

walker_mdh_hv = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["walker"]["MDH"].run_id}/{HV_FILE}.csv")
walker_mlp_hv = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["walker"]["MLP"].run_id}/{HV_FILE}.csv")

walker_mdh_sp = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["walker"]["MDH"].run_id}/{SP_FILE}.csv")
walker_mlp_sp = pd.read_csv(f"{BASE_FILEPATH}/{icra.FINAL_CONFIGS["walker"]["MLP"].run_id}/{SP_FILE}.csv")

ax3 = plt.subplot(2,1,2)
plot_hv(ax3, walker_mdh_hv["_step"], walker_mdh_hv[HV_FILE], label = "Hypernetwork", color = "#236E7D", marker='.')
plot_hv(ax3, walker_mlp_hv["_step"], walker_mlp_hv[HV_FILE], label = "MLP", color = "#9136AA", marker='.')
ax3.legend()
ax3.set_ylabel("Hypervolume")
ax3.set_xlabel("Training Steps")
codesign.dress_axis(ax3)
ax4 = ax3.twinx()
plot_hv(ax4, walker_mdh_sp["_step"], walker_mdh_sp[SP_FILE], label = "Hypernetwork", color = "#236E7D", linestyle = '--', marker='.')
plot_hv(ax4, walker_mlp_sp["_step"], walker_mlp_sp[SP_FILE], label = "MLP", color = "#9136AA", linestyle = '--', marker='.')
ax4.set_ylim([0, 0.5])
ax4.set_xlabel("Training Steps")
ax4.set_ylabel("Spacing")
codesign.dress_axis(ax4)


steps = cheetah_mdh_hv["_step"]
best_hnet_hvol = cheetah_mdh_hv[HV_FILE][steps < icra.MO_CHECKPOINT].max()
# best_hnet_hvol =  max(cheetah_mdh_hv[HV_FILE][steps < icra.MO_CHECKPOINT])
steps = cheetah_mlp_hv["_step"]
best_dmlp_hvol =  cheetah_mlp_hv[HV_FILE][steps < icra.MO_CHECKPOINT].max()
improvement = (best_hnet_hvol - best_dmlp_hvol)/best_dmlp_hvol
print(f"Hypernet Hypervolume is {improvement*100} better than MLP Hypervolume for Cheetah")

steps = walker_mdh_hv["_step"]
best_hnet_hvol =  max(walker_mdh_hv[HV_FILE][steps < icra.MO_CHECKPOINT])
best_dmlp_hvol =  max(walker_mlp_hv[HV_FILE][steps < icra.MO_CHECKPOINT])
improvement = (best_hnet_hvol - best_dmlp_hvol)/best_dmlp_hvol
print(f"Hypernet Hypervolume is {improvement*100} better than MLP Hypervolume for Walker")


plt.savefig(f"scripts/icra/outputs/training_curve.pdf")
plt.savefig(f"scripts/icra/outputs/training_curve.png", dpi=600)
plt.show()
