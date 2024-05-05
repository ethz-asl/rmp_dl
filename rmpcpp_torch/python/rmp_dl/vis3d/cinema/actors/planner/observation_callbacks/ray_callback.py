

import abc
from typing import Dict, List, Optional
import numpy as np
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import GeometryGetter, UpdateGeometries
from rmp_dl.vis3d.utils import Open3dUtils


class RayCallback(GeometryGetter):
    def __init__(self, maxnorm=None, mesh=False, mesh_radius=0.03):
        self.maxnorm = maxnorm
        self.mesh = mesh
        self.mesh_radius = mesh_radius

    def __call__(self, observation: Optional[Dict]) -> Optional[UpdateGeometries]:
        if observation is None:
            raise IndexError("Observation is None")
        endpoints = self._extract_ray_endpoints_from_observation(observation)

        if endpoints is None:
            return None

        pos = np.array(observation["state"]["pos"])

        geometry = Open3dUtils.get_rays_geometry(endpoints, pos, maxnorm=self.maxnorm)

        # The ray callback updates the rays for each step, so we add this geometry to both the 
        # to_add and to_remove lists. The to_add list is processed first, so the ray will be
        # added to the scene before an image is taken and it is removed.
        return UpdateGeometries(
            to_add=[geometry],
            to_remove=[geometry]
        )


    @abc.abstractmethod
    def _extract_ray_endpoints_from_observation(self, observation: Dict) -> Optional[np.ndarray]: ...


class UnprocessedRayCallback(RayCallback):
    def __init__(self, maxnorm=10.0): 
        super().__init__(maxnorm=maxnorm) # Makes the colouring of the rays consistent across frames

    def _extract_ray_endpoints_from_observation(self, observation: Dict) -> np.ndarray:
        rays = observation["rays"]["rays"]
        if self.maxnorm is not None:
            rays = np.clip(rays, -self.maxnorm, self.maxnorm)

        return rays
    
class RayDecoderOutputCallback(RayCallback):
    def __init__(self, softmax=True):
        super().__init__()
        self.softmax = softmax

    def _extract_ray_endpoints_from_observation(self, observation: Dict) -> np.ndarray:
        rays = observation["learned_policy"]["output_ray_predictions"]

        if self.softmax:
            rays = np.exp(rays)
            rays = rays / np.sum(rays)

        # Normalize between 0 and 1
        rays = (rays - np.min(rays)) / (np.max(rays) - np.min(rays))
        return rays
    

class RayTransition(RayCallback):
    def __init__(self, first: Optional[RayCallback], second: Optional[RayCallback], start: int, length: int):
        super().__init__()
        self.first = first
        self.second = second
        if first is None and second is None:
            raise ValueError("At least one of first and second must be non-None")
        self.start = start
        self.length = length
        self.end = start + length
        self.count = -1

    def _extract_ray_endpoints_from_observation(self, observation: Dict) -> Optional[np.ndarray]:
        self.count += 1
        if self.count < self.start or self.count >= self.end:
            return None

        first = None
        if self.first is not None:
            first = self.first._extract_ray_endpoints_from_observation(observation)
            if first is None: return None
        second = None
        if self.second is not None:
            second = self.second._extract_ray_endpoints_from_observation(observation)
            if second is None: return None

        # If either is still None, we create a zero array so we get a transition from nothing. 
        if first is None:
            first = np.zeros_like(second)
        if second is None:
            second = np.zeros_like(first)

        r = (self.count - self.start) / self.length

        return first * (1 - r) + second * r
