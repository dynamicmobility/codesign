"""Training / inference entry points for ``codesign`` policies.

``inference`` loads trained (MO) design-hypernetwork checkpoints back into callable
policies.
"""

from codesign.learning.inference import (
    load_design_hypernetwork,
    load_mo_design_hypernetwork,
)

__all__ = [
    "load_design_hypernetwork",
    "load_mo_design_hypernetwork",
]
