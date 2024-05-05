
import abc
import numpy as np

import rmp_dl.util.io as rmp_io
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen import ProbabilisticWorldgen
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase


class ProbabilisticWorldgenFactoryFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, num_obstacles, seed, start, goal) -> WorldgenBase: ... 

class SphereBoxWorld(ProbabilisticWorldgenFactoryFunction):
    def __call__(self, *args, **kwargs) -> WorldgenBase:
        return ProbabilisticWorldgenFactory.sphere_box_world(*args, **kwargs)

class PlaneWorld(ProbabilisticWorldgenFactoryFunction):
    def __call__(self, *args, **kwargs) -> WorldgenBase:
        return ProbabilisticWorldgenFactory.plane_world(*args, **kwargs)

class World25d(ProbabilisticWorldgenFactoryFunction):
    def __call__(self, *args, **kwargs) -> WorldgenBase:
        return ProbabilisticWorldgenFactory.world25d(*args, **kwargs)


class ProbabilisticWorldgenFactory:
    @staticmethod
    def sphere_box_world_without_start_goal(num_obstacles, 
                         seed, 
                         with_bounds=True,
                        ):

        _, world_config = rmp_io.ConfigUtil.get_general_and_world_configs()

        params = world_config["sphere_box_world"]
        params["obstacle_groups"][0]["count"] = num_obstacles

        worldgen = ProbabilisticWorldgen(**params, seed=seed, 
                                         forbidden_locations=[], forbidden_locations_margin=0.5).generate_world(with_bounds=with_bounds)
        
        return worldgen
    

    @staticmethod
    def sphere_box_world(num_obstacles, 
                         seed, 
                         start = np.array([1., 1., 1.]),
                         goal = np.array([9., 9., 9.]),
                         with_bounds=True,
                        ):

        _, world_config = rmp_io.ConfigUtil.get_general_and_world_configs()

        params = world_config["sphere_box_world"]
        params["obstacle_groups"][0]["count"] = num_obstacles

        worldgen = ProbabilisticWorldgen(**params, seed=seed, 
                                         forbidden_locations=[start, goal], forbidden_locations_margin=0.5).generate_world(with_bounds=with_bounds)
        worldgen.set_start(start)
        worldgen.set_goal(goal)
        
        return worldgen
    

    @staticmethod
    def plane_world(num_obstacles, 
                    seed, 
                    start = np.array([1., 1., 1.]),
                    goal = np.array([9., 9., 9.]),
                    with_bounds=True,
                    ):
        
        _, world_config = rmp_io.ConfigUtil.get_general_and_world_configs()

        params = world_config["plane_world"]
        params["obstacle_groups"][0]["count"] = num_obstacles

        worldgen = ProbabilisticWorldgen(**params, seed=seed, 
                                         forbidden_locations=[start, goal], forbidden_locations_margin=0.5).generate_world(with_bounds=with_bounds)
        worldgen.set_start(start)
        worldgen.set_goal(goal)
        
        return worldgen
    

    @staticmethod
    def world25d(num_obstacles,
                 seed, 
                 start = np.array([1., 2.5, 1.]),
                 goal = np.array([19., 2.5, 19.]),
                 ):
        
        _, world_config = rmp_io.ConfigUtil.get_general_and_world_configs()

        params = world_config["world25d"]
        params["obstacle_groups"][0]["count"] = num_obstacles

        worldgen = ProbabilisticWorldgen(**params, seed=seed, 
                                         forbidden_locations=[start, goal], forbidden_locations_margin=0.5).generate_world()
        worldgen.set_start(start)
        worldgen.set_goal(goal)
        
        return worldgen