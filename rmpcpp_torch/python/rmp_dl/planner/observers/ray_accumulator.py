
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import State
from rmp_dl.util.halton_sequence import HaltonUtils
import torch

class RayAccumulator(Observer):
    def __init__(self, steps: float, ray_observation_getter: Observer):
        super().__init__()
        self.steps = steps
        self.getter = ray_observation_getter
        self.rays = []

    def _get_observation(self, state: State) -> torch.Tensor:
        rays = self.getter(state)

        self.rays.append(rays)
        if len(self.rays) > self.steps:
            self.rays.pop(0)
        
        stacked = torch.vstack(self.rays)

        stacked_sum = torch.sum(stacked, dim=0)
        non_zero_sum = torch.sum(stacked != 0, dim=0)
        non_zero_sum[non_zero_sum == 0] = 1 # These don't matter but we don't want to divide by zero

        return stacked_sum / non_zero_sum