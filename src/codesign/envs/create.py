from codesign.envs.codesign_base import CodesignBase, CodesignMO2SO
from codesign.envs.cheetah import *
from codesign.envs.TwoAxis import TwoAxis
from codesign.envs.RHex import RHex
import minimal_mjx as mm

def load_env(config: dict, backend: str | None = None) -> tuple[CodesignBase, dict]:
    env_name = config['env']
    env_params = mm.create_config_dict(config['env_config'])
    if backend is None:
        backend = config['backend']

    if(env_name == "MOCodesignCheetah"):
        env = MOCodesignCheetah(env_params=env_params, backend=backend)
    elif(env_name == "MOCodesignCheetah1D"):
        env = MOCodesignCheetah1D(env_params=env_params, backend=backend)
    elif(env_name == 'MOCodesignCheetah1DOldEnergy'):
        env = MOCodesignCheetah1DOldEnergy(env_params=env_params, backend=backend)
    elif(env_name == 'MOCodesignCheetahBackLegs'):
        env = MOCodesignCheetahBackLegs(env_params=env_params, backend=backend)
    elif(env_name == "TwoAxis"):
        env = TwoAxis(env_params=env_params, backend=backend)
    elif(env_name == "RHex"):
        env = RHex(env_params=env_params, backend=backend)
    else:
        raise ValueError(f"Unknown env '{env_name}'")

    # Single Objective wrapper
    if(config['algorithm'] == "design_hypernetwork"):
        env = CodesignMO2SO(env, env_params.reward.optimization.default_scalarization)

    # Single Objective wrapper
    if(config['algorithm'] == "ppo"):
        env = CodesignMO2SO(env, env_params.reward.optimization.default_scalarization)

    return env, env_params