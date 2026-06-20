"""Design-construction helpers shared across training/eval."""

import numpy as np


def uniform_design_sweep(config, num_envs: int) -> np.ndarray:
    """Uniform sweep of ``num_envs`` designs spanning the configured design range.

    Returns an array of shape ``(num_envs, design_dim)`` in ``[design_low, design_high]``.
    """
    design = config["learning_params"]["design_params"]
    low = float(design["design_low"])
    high = float(design["design_high"])
    dim = int(design["design_dim"])
    return np.linspace(low, high, num_envs).reshape(num_envs, dim).astype(np.float32)
