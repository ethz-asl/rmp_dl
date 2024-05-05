

from typing import Dict, List, Optional
import numpy as np

from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import GeometryGetter, UpdateGeometries
from rmp_dl.vis3d.utils import Open3dUtils

import open3d as o3d
from rmp_dl.vis3d.vis3d import Plot3D

class TrajectoryCallback(GeometryGetter):
    def __init__(self, color: Optional[List[float]] = None, radius: float = 0.02):
        self.color = color
        self.radius = radius

    def __call__(self, observation: Optional[Dict]) -> UpdateGeometries:
        if observation is None:
            raise IndexError("Observation is None")
        pos = observation["state"]["pos"]

        if self.color is not None:
            c = self.color
        else:
            # If color is not given, color by velocity
            c = observation["state"]["vel"]
            c = Open3dUtils.get_RGB(np.array([c]))[0]
        
        pcd = Plot3D.get_sphere_geometry(pos, color=np.array(c), radius=self.radius)

        # We add a single point every time and don't remove the previous one
        # TODO: Maybe it is more performant to aggregate the points into a single pointcloud
        # object, and remove the old one. 
        return UpdateGeometries(
            to_add=[pcd],
            to_remove=[]
        )
        