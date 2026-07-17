from codesign.envs.CodesignBase import CodesignBase
from codesign.envs.CodesignCheetah import CodesignCheetah
from codesign.envs.TwoAxis import TwoAxis
from codesign.envs.RHex import RHex
import minimal_mjx as mm

def load_env(config: dict) -> CodesignBase:
    env_name = config['env']
    env_params = mm.create_config_dict(config['env_config'])
    backend = config['backend']
    
    if(env_name == "CodesignCheetah"):
        env = CodesignCheetah(env_params=env_params, backend=backend)
    elif(env_name == "TwoAxis"):
        env = TwoAxis(env_params=env_params, backend=backend)
    elif(env_name == "RHex"):
        env = RHex(env_params=env_params, backend=backend)
    else:
        raise ValueError(f"Unknown env '{env_name}'")

    return env, env_params