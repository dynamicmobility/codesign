"""Multi-objective JAX/MuJoCo environments.

Provides the registered MO-Playground environments (``MOCheetah``,
``MOHopper``, ``MOAnt``, ``MOWalker``, ``MOHumanoid``, ``NaviGait``) and
the base classes (``MultiObjectiveBase``, ``Multi2SingleObjective``)
they share. Use ``create_environment(config)`` to construct one from a
config file.
"""
from . import CodesignBase, CodesignCheetah, RHex

from codesign.envs.CodesignBase import CodesignBase as CodesignBase
from codesign.envs.CodesignBase import CodesignMO2SO
from codesign.envs.CodesignCheetah import CodesignCheetah as CodesignCheetah
from codesign.envs.RHex import RHex as RHex

__all__ = [
    "CodesignBase",
    "CodesignMO2SO",
    "CodesignCheetah",
    "RHex",
]