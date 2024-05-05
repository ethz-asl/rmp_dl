
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from rmp_dl.planner.planner_params import WorldgenSettings
from rmp_dl.worldgenpy.custom_worldgen import CustomWorldgen
from rmp_dl.worldgenpy.probabilistic_worldgen.obstacle_group import ObstacleGroup
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

class ProbabilisticWorldgen:
    def __init__(self, 
                 seed: Union[int, np.random.Generator], # You can also directly pass a generator
                 world_limits: Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]], # type: ignore
                 forbidden_locations: List[np.ndarray],
                 forbidden_locations_margin: float,
                 voxel_size: float,
                 voxel_truncation_distance_vox: float,
                 obstacle_groups: List[Dict[str, Any]]
                 ):
        world_limits: Tuple[np.ndarray, np.ndarray] = (np.array(world_limits[0]), np.array(world_limits[1]))
        self.world_limits = world_limits
        self.voxel_size = voxel_size
        self.voxel_truncation_distance_vox = voxel_truncation_distance_vox

        if isinstance(seed, int):
            self.generator = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self.generator = seed
        else:
            raise ValueError("Seed should be an int or a generator")

        self.forbidden_locations = forbidden_locations
        self.forbidden_locations_margin = forbidden_locations_margin

        self.obstacle_groups: List[ObstacleGroup] = []
        self._setup_groups(obstacle_groups)

    def _setup_groups(self, obstacle_groups):
        for group in obstacle_groups: 
            self.obstacle_groups.append(ObstacleGroup.resolve_group(**group, 
                                                                    generator=self.generator, 
                                                                    world_limits=self.world_limits, 
                                                                    forbidden_locations=self.forbidden_locations,
                                                                    forbidden_locations_margin=self.forbidden_locations_margin))

    def generate_world(self, with_bounds=True):
        custom_worldgen = CustomWorldgen(WorldgenSettings(self.world_limits, self.voxel_size, self.voxel_truncation_distance_vox))
        for group in self.obstacle_groups: 
            group.put_obstacles(custom_worldgen)
        if with_bounds:
            custom_worldgen.add_bounds_automatic()
        return custom_worldgen 
     
    @staticmethod
    def get_number_of_obstacles_from_groups(obstacle_groups: List[Dict[str, Any]]):
        return sum([group["count"] for group in obstacle_groups])