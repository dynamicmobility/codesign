from moplayground.envs.generic import MultiObjectiveBase
from ml_collections import config_dict
from pathlib import Path
from abc import abstractmethod
from mujoco import mjx
from mujoco_playground._src import mjx_env
import jax


class CodesignInterface(MultiObjectiveBase):

    @abstractmethod
    def generate_model(self, d) -> mjx.Model:
        """Given a design parameter d, generate a new mj/mjx Model"""

    @abstractmethod
    def resample_design(self, rng):
        """Given an rng, return a new design parameter"""