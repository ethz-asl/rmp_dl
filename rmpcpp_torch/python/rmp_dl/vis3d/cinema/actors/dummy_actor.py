
from typing import List
from rmp_dl.vis3d.cinema.actor import ActorBase

import open3d as o3d

class DummyActor(ActorBase):
    def __init__(self, steps: int):
        self.steps = steps

    def has_next_step(self) -> bool:
        return self.steps > 0
    
    def next_step(self) -> None:
        self.steps -= 1

    def get_geometries_to_remove(self) -> List[o3d.geometry.Geometry]:
        return []
    
    def get_geometries_to_add(self) -> List[o3d.geometry.Geometry]:
        return []
    