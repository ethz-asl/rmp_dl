
from typing import Callable, Dict, Optional
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import State
from rmp_dl.util.halton_sequence import HaltonUtils
import torch

class RayObservationInterpolator(Observer):
    def __init__(self, 
                 num_rays: int,
                 ray_observation_getter: Observer, 
                 iterations: int = 0,
                 kernel_size: int = 5,
                 save_intermediate_rays: int = 0, 
                 observation_callback: Optional[Callable[[str, Dict], None]] = None,
                 callback_name: Optional[str] = None,
                 ):
        super().__init__()
        self.ray_observation_getter = ray_observation_getter
        self.num_rays = num_rays
        self.iterations = iterations
        self.kernel_size = kernel_size

        if save_intermediate_rays and (observation_callback is None or callback_name is None):
            raise ValueError("If save_intermediate_rays is True, observation_callback and callback_name must be provided")

        self.save_intermediate_rays = save_intermediate_rays
        self.observation_callback = observation_callback
        self.callback_name = callback_name

        self._nearest_neighbor_indices = self._get_nearest_neighbor_indices(num_rays)  # (N, N)
        self._nearest_neighbor_indicesk = self._nearest_neighbor_indices[:, :kernel_size]  # (N, k)


    def _get_observation(self, state: State) -> torch.Tensor:
        rays = self.ray_observation_getter(state)
        original_rays = rays.clone()
        # First we set all unset rays to the value of the nearest neighbor
        rays = self._nearest_neighbor_interpolation(rays)

        # Then we smooth the rays with kernel size
        rays = self._nearest_neighbor_smoothing(rays, original_rays)

        return rays

    def _get_nearest_neighbor_indices(self, num_rays):
        # First we compute the endpoints of the rays
        points = HaltonUtils.get_ray_endpoints_from_halton_distances(np.ones(num_rays))
        points = torch.from_numpy(points).to(torch.float32).to(torch.device("cuda")) # (N, 3)

        # Then for each ray_endpoint we find the k nearest neighbors
        
        # We dot product the points with themselves to get the nearest neighbors
        dot_products = points @ points.T # (N, N)

        # We then sort the dot products to get the indices of the nearest neighbors
        _, nearest_neighbors_indices = torch.topk(dot_products, num_rays, dim=1, largest=True, sorted=True) # (N, N)

        return nearest_neighbors_indices


    def _nearest_neighbor_interpolation(self, rays: torch.Tensor) -> torch.Tensor:
        # We find the first set nearest neighbor for each ray
        nearest_neighbor_values = rays[self._nearest_neighbor_indices] # (N, N)

        # We basically want to find the first entry that is not 0 for each ray (row)
        # We do this by first converting the tensor to indicator values, doing a cumulative sum
        # and then doing an and of the indicator values and cumsum==1
        indicators = torch.zeros_like(nearest_neighbor_values, device="cuda")
        indicators[nearest_neighbor_values > 0] = 1
        cumsum = torch.cumsum(indicators, dim=1)
        first_non_zero = torch.logical_and(cumsum == 1, indicators)

        new_rays = torch.masked_select(nearest_neighbor_values, first_non_zero)

        return new_rays
        
    def _nearest_neighbor_smoothing(self, rays: torch.Tensor, original_rays: torch.Tensor) -> torch.Tensor:
        intermediate_saves = [rays.cpu().numpy()]

        # We do N iterations of averaging over the nearest K neighbors
        for i in range(self.iterations):
            rays = self._nearest_neighbor_smoothing_iteration(rays, original_rays)

            if self.save_intermediate_rays > 0 and i % self.save_intermediate_rays == 0:
                intermediate_saves.append(rays.cpu().numpy())
        if self.save_intermediate_rays > 0:
            self.observation_callback(self.callback_name, {f"ray_list": intermediate_saves})

        return rays
    
    def _nearest_neighbor_smoothing_iteration(self, rays: torch.Tensor, original_rays: torch.Tensor) -> torch.Tensor:
        # We first gather the nearest neighbors for each ray
        nearest_neighbor_rays = rays[self._nearest_neighbor_indicesk] # (N, k)

        # Then we average over the nearest neighbors
        new_rays = torch.mean(nearest_neighbor_rays, dim=1)
        
        # Rays that were originally set should always maintain their own value
        new_rays[original_rays > 0] = original_rays[original_rays > 0]

        return new_rays