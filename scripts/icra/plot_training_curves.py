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

HNET_FILENAME = "scripts/icra/outputs/eywf0dij/wandb_export_2026-09-12T15_59_22.066-04_00.csv"
DMLP_FILENAME = "scripts/icra/outputs/dkfq4g7t/wandb_export_2026-09-12T15_58_08.087-04_00.csv"

# Load the CSVs in "scripts/icra/outputs/{RUN_ID...}/" into pandas dataframes
hnet_data =pd.read_csv(HNET_FILENAME)
dmlp_data =pd.read_csv(DMLP_FILENAME)


fig, ax = plt.subplots()

data_key = next(key for key in hnet_data.keys() if key != "Step")
ax.plot(hnet_data["Step"], hnet_data[data_key], label="Hypernetwork Policy")
best_hnet_hvol = max(hnet_data[data_key])

data_key = next(key for key in dmlp_data.keys() if key != "Step")
ax.plot(dmlp_data["Step"], dmlp_data[data_key], label="MLP Policy")
best_dmlp_hvol = max(dmlp_data[data_key])

ax.legend()

ax.set_xlabel("Training Step")
ax.set_ylabel("Design Pareto Hypervolume")

improvement = (best_hnet_hvol - best_dmlp_hvol)/best_dmlp_hvol
print(f"Hypernet Hypervolume is {improvement*100} better than MLP Hypervolume")

codesign.dress_axis(ax)

plt.savefig(f"scripts/icra/outputs/training_curve.pdf")
plt.savefig(f"scripts/icra/outputs/training_curve.png", dpi=600)
plt.show()
