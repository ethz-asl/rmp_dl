
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import State
from rmp_dl.util.halton_sequence import HaltonUtils
import torch

class RayObservationCombiner(Observer):
    def __init__(self, method="max", **getters):
        super().__init__()
        self.getters = getters
        self.method=method

    def _get_observation(self, state: State) -> torch.Tensor:
        rays = [getter(state) for getter in self.getters.values()]

        rays = torch.vstack(rays).T # (num_rays, num_getters)

        # We assume that the getters set 0 for rays that are not observed

        # We don't just want to sum them together, as if there is overlap between observers, this results in weird values
        if self.method == "max":
            rays = torch.max(rays, dim=1).values
        elif self.method == "min":
            # First we have to set all 0s to inf, so that they don't get picked as the min
            rays[rays == 0] = float("inf")
            rays = torch.min(rays, dim=1).values
            # Then we set the inf back to 0
            rays[rays == float("inf")] = 0
        else:
            raise ValueError(f"Unknown method {self.method}")

        return rays
