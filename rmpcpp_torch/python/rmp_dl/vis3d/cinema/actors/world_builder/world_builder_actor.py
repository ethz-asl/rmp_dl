

from typing import Callable, Iterator, List
from rmp_dl.vis3d.cinema.actor import ActorBase

import open3d as o3d
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase


class WorldBuilderActor(ActorBase):
    def __init__(self, worldgen_callable: Callable[[int], WorldgenBase], num_obstacles: int):
        """Builds up a world with a number of obstacles by rebuilding the world with 1 new obstacle everytime 
        and rerendering. 

        Args:
            worldgen_callable (Callable[[int], WorldgenBase]): Callable that returns worldgen object, input is the number of obstacles
            num_obstacles (int): maximum number of obstacles we are building to
        """
        self.worldgen_callable = worldgen_callable
        self.num_obstacles = num_obstacles
        self.count = 0
        self.worldgen = None

    def has_next_step(self) -> bool: 
        return self.count <= self.num_obstacles

    def next_step(self) -> None: 
        self.worldgen = self.worldgen_callable(self.count)
        self.count += 1
        self.world_geometry = Plot3D.get_world_geometry(self.worldgen)

    def get_geometries_to_remove(self) -> List[o3d.geometry.Geometry]:
        return [self.world_geometry]

    def get_geometries_to_add(self) -> List[o3d.geometry.Geometry]:
        return [self.world_geometry]


    