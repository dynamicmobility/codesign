# Loads in a config file for a hypernetwork and uses it to compute the size of an equivalent MLP network.

import os
import argparse
import functools
import time
from pathlib import Path

import minimal_mjx as mm
import moplayground as mop
import numpy as np
import codesign

CONFIG_PATH = "config/design_hypernetwork/cheetah6D.yaml"

def num_parameters_per_layer(num_in: int, num_out: int):
    return num_in*num_out + num_out

def num_parameters_in_network(network: list):
    return sum([num_parameters_per_layer(network[i-1], network[i]) for i in range(1, len(network))])

def equivalent_2_layer_net(num_params:int, input_size: int, output_size:int):
    b = input_size + output_size + 2
    a = 1
    c = output_size-num_params
    return max((-b+np.sqrt(b**2 - 4*a*c))/(2*a), (-b-np.sqrt(b**2 - 4*a*c))/(2*a))

def main(config_path: str,):
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env, _ = codesign.load_env(config, backend = 'jnp')

    hypersize = config.learning_params.network_params.hypersize
    if(config.algorithm in ("design_hypernetwork", "design_lookup_hypernetwork")):
        inputs = len(config.env_config.codesign.default_design)
    elif(config.algorithm == ("mo_design_hypernetwork", "mo_design_predictor_hypernetwork")):
        inputs = len(config.env_config.codesign.default_design) + len(config.env_config.reward.objectives)

    encoder_outputs = config.learning_params.network_params.num_features 

    base_policy_size = config.learning_params.network_params.policy_hidden_layer_sizes
    base_network_desc = [env.observation_size["state"][-1]] + base_policy_size + [env.action_size]


    num_base_network_policy_params = num_parameters_in_network(base_network_desc)

    hypernet_desc = [inputs] + hypersize + [encoder_outputs] + [num_base_network_policy_params]
    num_params = num_parameters_in_network(hypernet_desc)
    eq_2_layer = equivalent_2_layer_net(num_params, env.observation_size["state"][-1], env.action_size) 
    print(f"There are {num_base_network_policy_params} params in the base policy network")
    print(f"There are {num_params} params in the hypernetwork")
    print(f"An equivalent design MLP would have {eq_2_layer} params")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    args = parser.parse_args()
    main(
        config_path=args.config
    )