"""Cheetah environment."""

from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from moplayground.envs.generic.mobase import MultiObjectiveBase
from moplayground.envs.dmcontrol.interface import CheetahInterface
from pathlib import Path

# Generate a Cheetah 