
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import State
from rmp_dl.util.halton_sequence import HaltonUtils
import torch

class RayStochasticDownsampler(Observer):
    def __init__(self, fraction: float, ray_observation_getter: Observer):
        super().__init__()
        self.fraction = fraction
        self.getter = ray_observation_getter

    def _get_observation(self, state: State) -> torch.Tensor:
        rays = self.getter(state)

        # Subsample a fraction of the rays
        unobserved_mask = torch.rand(rays.shape) > self.fraction
        
        rays[unobserved_mask] = 0
        return rays

