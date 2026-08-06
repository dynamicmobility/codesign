from .base import Weights, Robot, Codesign

class CheetahWeights(Weights):
    run       : float
    energy    : float
    height    : float
    alive     : float
    done      : float

class Cheetah(Robot):
    ground_margin: float | None = None
    codesign: Codesign