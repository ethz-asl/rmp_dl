from typing import Callable, Iterator, List
from rmp_dl.vis3d.arrow import get_arrow_geometry
from rmp_dl.vis3d.cinema.actor import ActorBase

import open3d as o3d
from rmp_dl.vis3d.geodesic_vis import DenseSampledVis
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase


class WorldSamplerActor(ActorBase):
    def __init__(self, worldgen, num_arrows, scale):
        """Builds up a world with a number of obstacles by rebuilding the world with 1 new obstacle everytime 
        and rerendering. 

        Args:
            worldgen_callable (Callable[[int], WorldgenBase]): Callable that returns worldgen object, input is the number of obstacles
            num_obstacles (int): maximum number of obstacles we are building to
        """
        self.count = -1
        self.scale = scale
        self.observations = list(DenseSampledVis.get_observations(worldgen, num_arrows))

    def has_next_step(self) -> bool: 
        return self.count < len(self.observations) - 1

    def next_step(self) -> None: 
        self.count += 1

    def get_geometries_to_remove(self) -> List[o3d.geometry.Geometry]:
        return []

    def get_geometries_to_add(self) -> List[o3d.geometry.Geometry]:
        observation = self.observations[self.count]
        arrow = get_arrow_geometry(observation["state"]["pos"], observation["expert_policy"]["geodesic"], scale=self.scale)
        arrow.paint_uniform_color([1, 0, 0])
        return [arrow]


    