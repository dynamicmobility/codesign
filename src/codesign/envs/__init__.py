"""Multi-objective JAX/MuJoCo environments.

Provides the registered MO-Playground environments (``MOCheetah``,
``MOHopper``, ``MOAnt``, ``MOWalker``, ``MOHumanoid``, ``NaviGait``) and
the base classes (``MultiObjectiveBase``, ``Multi2SingleObjective``)
they share. Use ``create_environment(config)`` to construct one from a
config file.
"""
from . import CodesignBase, CodesignCheetah
from .EnvLoader import load_env, RHex

from codesign.envs.CodesignBase import CodesignBase, MOCodesignBase, CodesignMO2SO, Codesign2SingleDesign
from codesign.envs.CodesignCheetah import MOCodesignCheetah
from codesign.envs.RHex import RHex as RHex

__all__ = [
    "CodesignBase",
    "CodesignMO2SO",
    "CodesignCheetah",
    "MOCodesignBase",
    "RHex",
    "load_env",
    "Codesign2SingleDesign",
    "MOCodesignCheetah"
]