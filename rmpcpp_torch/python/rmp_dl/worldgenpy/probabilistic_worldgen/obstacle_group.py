from __future__ import annotations

import abc
import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rmp_dl.worldgenpy.custom_worldgen import CustomWorldgen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_primitive import ProbabilisticPrimitive

from nvbloxSceneBindings import NvbloxPrimitive

class ObstacleGroup(abc.ABC):
    def __init__(self, 
                 name,
                 generator: np.random.Generator,
                 count: int, 
                 world_limits: Tuple[np.ndarray, np.ndarray],
                 forbidden_locations: List[np.ndarray],
                 forbidden_locations_margin: float,
                 ):
        self.name = name
        self.count = count
        self.generator = generator
        self.world_limits = world_limits
        self.forbidden_locations = forbidden_locations
        self.forbidden_locations_margin = forbidden_locations_margin

    @abc.abstractmethod
    def _sample_primitive(self) -> NvbloxPrimitive: ...

    def put_obstacles(self, custom_world: CustomWorldgen): 
        """
        Put obstacles from the group according to its specification into the world
        """
        count = 0
        while True:
            if (count := count + 1) > self.count:
                break

            primitive = self._sample_primitive()

            continue_flag = False
            for location in self.forbidden_locations:
                if primitive.get_distance_to_point(location) < self.forbidden_locations_margin:
                    continue_flag = True
                    break
            if continue_flag:
                continue
            
            custom_world.add_primitive(primitive)

    @staticmethod
    def resolve_group(group_type, **kwargs) -> ObstacleGroup:
        if group_type == "weighted":
            return WeightedGroup(**kwargs)
        else: 
            raise ValueError(f"Unknown group type {group_type}")


class WeightedGroup(ObstacleGroup):
    def __init__(self, 
                 obstacles: List[Dict[str, Any]],
                 weights: Optional[List[float]] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.weights = weights if weights is not None else [1] * len(obstacles)
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
        self.primitives: List[ProbabilisticPrimitive] = []

        for obstacle in obstacles:
            obstaclenew = copy.deepcopy(obstacle) # Don't want to alter the old dict # TODO: Fix that this is necessary
            primitive_type = obstaclenew.pop("obstacle_type")
            
            primitive = ProbabilisticPrimitive.resolve_obstacle(primitive_type)(**obstaclenew, generator=self.generator, world_limits=self.world_limits)
            self.primitives.append(primitive)
        

    def _sample_primitive(self) -> NvbloxPrimitive:
        index = self.generator.choice(list(range(len(self.primitives))), p=self.weights)
        return self.primitives[index].get()