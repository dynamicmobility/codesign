# Loads in a config file for a hypernetwork and uses it to compute the size of an equivalent MLP network.

import argparse
import itertools
import math

import minimal_mjx as mm
import moplayground as mop
import numpy as np
import codesign

CONFIG_PATH = "config/design_hypernetwork/cheetah6D.yaml"
MAX_CANDIDATES = 1_000_000
MAX_BAND_FRACTION = 0.1
NUM_BANDS = 12

def num_parameters_per_layer(num_in: int, num_out: int):
    return num_in*num_out + num_out

def num_parameters_in_network(network: list):
    return sum([num_parameters_per_layer(network[i-1], network[i]) for i in range(1, len(network))])

def uniform_width(num_params: int, input_size: int, output_size: int, num_layers: int):
    """Real-valued hidden width h that hits ``num_params`` when every layer is width h.

    Such a network has (L-1)h^2 + (input + output + L)h + output parameters, so h is
    the positive root of that quadratic (linear when L == 1).
    """
    roots = np.roots([num_layers - 1, input_size + output_size + num_layers, output_size - num_params])
    return max([r.real for r in roots if abs(r.imag) < 1e-9 and r.real >= 1], default=1.0)

def equivalent_net(num_params: int, input_size: int, output_size: int, num_layers: int):
    """Widths of the ``num_layers``-deep MLP whose parameter count is closest to ``num_params``.

    Hidden widths are non-increasing, so the narrow layers sit at the output end. All
    but the last are enumerated within a band around the uniform-width solution; with
    those fixed the parameter count is linear in the last width, so that one is solved
    for directly. Bands widen only until the count is matched exactly, keeping the
    result as close to uniform as the arithmetic allows. Returns
    ``[input_size, *hidden, output_size]``.
    """
    uniform = int(round(uniform_width(num_params, input_size, output_size, num_layers)))
    if num_layers == 1:
        return [input_size, max(1, round((num_params - output_size)/(input_size + 1 + output_size))), output_size]

    best = None
    for band in np.unique(np.geomspace(1, max(1, MAX_BAND_FRACTION*uniform), NUM_BANDS).round()).astype(int):
        lo, hi = max(1, uniform - band), uniform + band
        if math.comb(hi - lo + num_layers - 1, num_layers - 1) > MAX_CANDIDATES:
            break
        # Non-increasing widths are combinations with replacement, read back to front.
        head = np.array([c[::-1] for c in itertools.combinations_with_replacement(range(lo, hi + 1), num_layers - 1)])
        fixed = input_size*head[:, 0] + head.sum(1) + ((head[:, :-1]*head[:, 1:]).sum(1) if num_layers > 2 else 0)
        # Every term in the last width shares the factor below, so invert for it.
        slope = head[:, -1] + 1 + output_size
        last = np.clip(np.round((num_params - fixed - output_size)/slope).astype(int), 1, head[:, -1])
        errors = np.abs(fixed + slope*last + output_size - num_params)
        closest = np.flatnonzero(errors == errors.min())
        # Ties go to the most front-loaded network, i.e. the lexicographically largest.
        pick = closest[np.lexsort(np.column_stack([head, last])[closest].T[::-1])[-1]]
        if best is None or errors[pick] < best[1]:
            best = (head[pick].tolist() + [int(last[pick])], errors[pick])
        if best[1] == 0:
            break
    return [input_size] + best[0] + [output_size]

def main(config_path: str, layers: list):
    config = mm.utils.config.create_config_dict(mop.utils.read_config(config_path))
    env, _ = codesign.load_env(config, backend = 'jnp')

    hypersize = list(config.learning_params.network_params.hypersize)
    if(config.algorithm in ("design_hypernetwork", "design_lookup_hypernetwork")):
        inputs = len(config.env_config.codesign.default_design)
    elif(config.algorithm in ("mo_design_hypernetwork", "mo_design_predictor_hypernetwork")):
        inputs = len(config.env_config.codesign.default_design) + len(config.env_config.reward.optimization.objectives)
    else:
        raise Exception(f'Unknown algorithm {config.algorithm}')

    encoder_outputs = config.learning_params.network_params.num_features

    obs_size = env.observation_size["state"][-1]
    base_policy_size = list(config.learning_params.network_params.policy_hidden_layer_sizes)
    base_network_desc = [obs_size] + base_policy_size + [env.action_size]
    num_base_network_policy_params = num_parameters_in_network(base_network_desc)

    hypernet_desc = [inputs] + hypersize + [encoder_outputs] + [num_base_network_policy_params]
    num_params = num_parameters_in_network(hypernet_desc)

    # A design-conditioned MLP reads the design (and, when multi-objective, the
    # preference weights) alongside the state, so its input is that much wider.
    mlp_inputs = obs_size + inputs

    print(f"base policy  : {base_network_desc} -> {num_base_network_policy_params:,} params")
    print(f"hypernetwork : {hypernet_desc} -> {num_params:,} params")
    print(f"\nequivalent design MLP ({obs_size} state + {inputs} design/objective inputs, "
          f"target {num_params:,} params):")

    rows = []
    for num_layers in layers:
        desc = equivalent_net(num_params, mlp_inputs, env.action_size, num_layers)
        total = num_parameters_in_network(desc)
        rows.append((f"L={num_layers}", str(desc), f"{total:,} params", f"({total - num_params:+,})"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()
    main(
        config_path=args.config,
        layers=args.layers,
    )
