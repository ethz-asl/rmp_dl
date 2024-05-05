
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional
import numpy as np
from rmp_dl.learning.lightning_module import RayLightningModule
from rmp_dl.planner.planner_params import RayObserverParameters
from rmp_dl.planner.observers.ray_observer import RayObserver
from rmp_dl.vis3d.cinema.actor import ActorBase

import open3d as o3d
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase
import torch

@dataclass
class WorldDescription():
    worldgen: Optional[WorldgenBase]
    goal_location: Optional[np.ndarray]
    position: Optional[np.ndarray]

    def __iter__(self):
        return iter((self.worldgen, self.goal_location, self.position))

class StaticPredictorActor(ActorBase):
    def __init__(self, description_iterator: Iterator[WorldDescription], model: RayLightningModule, **kwargs):
        """Visualizes the output of a model that is static:
        Instead of the model doing a rollout, we vary the world, goal location and location of the model via iterators.
        We iterate over the WorldDescription class. The first world description should have all values defined. After that, 
        they can be none as well, which means that the the last geometry corresponding to that object will be reused.
        """
        super().__init__(**kwargs)
        self.description_iterator = description_iterator
        self.next_description: WorldDescription = None # type: ignore --> Will be set in has_next_step, or we exit if no next step
        self.worldgen: Optional[WorldgenBase] = None
        self.world_geometry = None
        self.goal_location: Optional[np.ndarray] = None
        self.goal_geometry = None
        self.position: Optional[np.ndarray] = None
        self.ray_geometry = None
        self.model = model
        self.ray_params = RayObserverParameters.from_yaml_general_config()

    def has_next_step(self) -> bool: 
        try:
            self.next_description = next(self.description_iterator)
            return True
        except StopIteration:
            return False

    def next_step(self) -> None: 
        worldgen, goal_location, position = self.next_description
        if worldgen is not None:
            self.worldgen = worldgen
            self.world_geometry = Plot3D.get_world_geometry(worldgen)
        if goal_location is not None:
            self.goal_location = goal_location
            self.goal_geometry = Plot3D.get_sphere_geometry(goal_location, [0, 1, 0], radius=0.1)
        if position is not None:
            self.position = position

        # If any of the information is None at this point, we raise an exception
        if self.worldgen is None or self.goal_location is None or self.position is None:
            raise ValueError("One of the world, goal or position is None. This is not allowed.")

        # We predict the rays
        rel_pos = self.goal_location - self.position
        vel = np.zeros_like(rel_pos)
        rel_pos = torch.from_numpy(rel_pos).float().to(device="cuda:0").detach()
        vel = torch.from_numpy(vel.copy()).float().to(device="cuda:0").detach()
        rays = RayObserver.get_rays(self.worldgen.get_tsdf(), self.position, self.ray_params).detach()

        pred = self.model(rays.unsqueeze(0).unsqueeze(0), 
                                rel_pos.unsqueeze(0).unsqueeze(0), 
                                vel.unsqueeze(0).unsqueeze(0)).squeeze().cpu().detach().numpy()

        pred = np.exp(pred)
        pred = pred / np.sum(pred)
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        self.ray_geometry = Open3dUtils.get_rays_geometry(pred, self.position)

    def get_geometries_to_remove(self) -> List[o3d.geometry.Geometry]:
        return [self.world_geometry, self.ray_geometry, self.goal_geometry]

    def get_geometries_to_add(self) -> List[o3d.geometry.Geometry]:
        return [self.world_geometry, self.ray_geometry, self.goal_geometry]


    