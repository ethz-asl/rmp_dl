from typing import Callable, Dict, Optional
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import PyState, State
from rmp_dl.radar_data.radar_data import RadarData
from rmp_dl.util.halton_sequence import HaltonUtils
import torch

class RadarPointsRayConverter(Observer):
    def __init__(self, 
                num_rays: int,
                observation_getter: Observer,):
        super().__init__()
        self._observation_getter = observation_getter
        self._points = []
        
        self.num_rays = num_rays
        points = HaltonUtils.get_ray_endpoints_from_halton_distances(np.ones(num_rays))
        self.endpoints = torch.from_numpy(points).to(torch.float32).to(torch.device("cuda")) # (N, 3
    
    def _get_observation(self, state: State) -> torch.Tensor:
        points = self._observation_getter(state) # (P, 3)

        # Transform into body frame
        pos = torch.from_numpy(state.pos).to(torch.float32).to(torch.device("cuda")) # (3,)
        points = points - pos # (P, 3)
        points_norms = torch.norm(points, dim=1, keepdim=True) # (P, 1)

        unit_vectors = points / points_norms # (P, 3)

        # We dot product all the point vectors with the ray endpoints to get the nearest ray for each point
        dot_products = self.endpoints @ unit_vectors.T  # (N, P)

        # We then find the argmax of the dot products to get the nearest ray for each point
        _, nearest_ray_indices = torch.max(dot_products, dim=0) # (P,)

        # We set the rays to the nearest ray for each point
        # An interesting (intended) side effect of the operation below is that 
        # if two points are closest to the same ray, the second point will overwrite the first point
        # This is intended as we want to keep the most recent point for each ray
        rays = torch.zeros(self.num_rays).to(torch.float32).to(torch.device("cuda"))
        rays[nearest_ray_indices] = points_norms.flatten()

        return rays
        
