from __future__ import annotations


from typing import Dict, List, Tuple
import numpy as np
from rmp_dl.learning.data.pipeline.nodes.worlds.single_world_base import SingleWorldBase
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor
from rmp_dl.planner.planner_params import WorldgenSettings
from rmp_dl.worldgenpy.custom_worldgen import CustomWorldgen
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

class ManualCubeWorldConstructor(WorldConstructor):
    def __init__(self, cubes, start, goal, **kwargs):
        super().__init__(**kwargs)
        self.cubes = cubes
        self.start = start
        self.goal = goal    
        
        # Early detect if there are issues with computing the hash
        self.__hash__()
        
    def __call__(self, *, continued_generation=False) -> WorldgenBase:
        if continued_generation: 
            # We don't support continued generation for this world as it is deterministic
            raise ValueError("Continued generation is not supported for ManualCubeWorld")
        
        settings = WorldgenSettings(self.world_limits, self.voxel_size, self.voxel_truncation_distance_vox)
        worldgen = CustomWorldgen(settings)

        for cube in self.cubes:
            worldgen.add_cube(np.array(cube["location"]), np.array(cube["size"]))

        worldgen.set_start(np.array(self.start))
        worldgen.set_goal(np.array(self.goal))
        worldgen.add_bounds_automatic()
        return worldgen
    
    def __str__(self) -> str:
        return f"ManualCubeWorld: wl={self.world_limits} c={self.cubes}, s={self.start}, g={self.goal}"
            
    def _get_world_equality_hashable_params(self) -> Tuple:
        params = (
            self.to_hashable(self.cubes),
        )
        return super()._get_world_equality_hashable_params() + params
    

class ManualCubeWorld(SingleWorldBase):
    def __init__(self,
                    cubes: List[Dict[str, List[float]]],  # List of cubes, each cube is a dict with keys "location" and "size"
                    start: List[float],
                    goal: List[float],
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cubes = cubes
        self.start = start
        self.goal = goal
            
    def get_world_constructor(self) -> WorldConstructor:
        return ManualCubeWorldConstructor(cubes=self.cubes, start=self.start, goal=self.goal, 
                                          **self.get_world_information_params())
