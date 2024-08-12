from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
import numpy as np
from rmp_dl.learning.data.pipeline.nodes.worlds.single_world_base import SingleWorldBase
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.seed_manager import WorldgenSeedManager
from rmp_dl.learning.data.utils.location_sampler import StartGoalMinDistSampler, UniformWorldLimitsSampler
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen import ProbabilisticWorldgen
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

class ProbabilisticWorldConstructor(WorldConstructor):
    def __init__(self, goal, start, seed, start_goal_margin_to_obstacles, obstacle_groups,
                 **kwargs):
        super().__init__(**kwargs)
        self.goal = goal
        self.start = start
        self.seed = seed
        self.start_goal_margin_to_obstacles = start_goal_margin_to_obstacles
        self.obstacle_groups = obstacle_groups
        
        self.num_obstacles = ProbabilisticWorldgen.get_number_of_obstacles_from_groups(self.obstacle_groups)
        
        # Generator state before the last call to generate_world
        self.probabilistic_worldgen_generator_prev = None
        # Generator state after the last call to generate_world
        self.probabilistic_worldgen_generator_cur = None
        
        # Early detect if there are issues with computing the hash 
        self.__hash__()    

    def __call__(self, *, continued_generation=False) -> WorldgenBase:
        """Generate the world. By setting continued_generation to True, the world will be regenerated without 
        resetting the underlying generating object, thus resulting in a new world.
        I.e. successive calls to __call__ with continued_generation=True will result in different worlds. 
        Successive calls to __call__ with continued_generation=False will result in the same world as last time.
        Note that if you call: 
        w1 = __call__(continued_generation=True/False (doesn't matter for first call))
        w2 = __call__(continued_generation=True)
        w3 = __call__(continued_generation=False)
        w4 = __call__(continued_generation=True)
        w2 will be identical to w3
        I.e. when setting continued generation to false, it will keep returning the same world as last call until you set it to true again.
        
        A bit hacky all of this, at first the whole point of this class was that it was identical on successive calls, 
        but later on the addition was made to regenerate worlds if they don't satisfy certain conditions, so this had to be added. 
        But I still want that successive calls to this function can return identical worlds, so now we have this mix. 
        We don't simply cache the generated world, as we want this function
        to be pickleable and maintain the same behavior when pickled and unpickled.
        (the worldgen object can't be pickled as it contains c++ objects)
        """
        self._set_probabilistic_worldgen(continued_generation)
        
        # We save the states before and after the call, so we can regenerate the same world if we want to
        self.probabilistic_worldgen_generator_prev = self.probabilistic_worldgen.generator
        _generated_world = self.probabilistic_worldgen.generate_world()
        self.probabilistic_worldgen_generator_cur = self.probabilistic_worldgen.generator

        # Start and goal have already been given to the random world generator 
        # which makes sure that they are located in free space
        _generated_world.set_goal(self.goal)
        _generated_world.set_start(self.start)
        return _generated_world

    def __str__(self) -> str:
        group_names = [group["name"] for group in self.obstacle_groups]
        
        return f"ProbabilisticWorld(no={self.num_obstacles}, s={self.seed}, group_names={group_names}, infl={self.inflation})"
    
    def _set_probabilistic_worldgen(self, continued_generation):
        if self.probabilistic_worldgen_generator_prev is None:
            # First call to generate world, use the seed
            seed_or_generator = self.seed
        elif not continued_generation:
            # If we are continuing the generation, we keep the same generator from before the last call
            # (so we get an identical world)
            seed_or_generator = self.probabilistic_worldgen_generator_prev
        else:
            # If we are not continuing, we take the generator state from after the last call
            # (which means that we will get a new world)
            seed_or_generator = self.probabilistic_worldgen_generator_cur
        
        self.probabilistic_worldgen = ProbabilisticWorldgen(
            seed=seed_or_generator,
            world_limits=self.world_limits,
            forbidden_locations=[self.start, self.goal],
            forbidden_locations_margin=self.start_goal_margin_to_obstacles,
            voxel_size=self.voxel_size,
            voxel_truncation_distance_vox=self.voxel_truncation_distance_vox,
            obstacle_groups=self.obstacle_groups,
        )

    def _get_world_equality_hashable_params(self) -> Tuple:
        params = (
            self.num_obstacles, 
            self.seed, 
            self.to_hashable(self.start), 
            self.to_hashable(self.goal),
            self.start_goal_margin_to_obstacles,
            self.to_hashable(self.obstacle_groups)
        )

        return super()._get_world_equality_hashable_params() + params


class ProbabilisticWorld(SingleWorldBase):
    """Probabilistic world sampler

    BE VERY CAREFUL WITH REFACTORING THIS CLASS. THE SEED GETS SET BY THE BASE CLASS 
    BEFORE EVERY CALL TO GET_WORLD_CONSTRUCTOR. MAKE SURE TO RUN WorldgenManager.get_seed 
    BEFORE DOING ANY CALCULATING WITH THE SEED. DO NOT JUST ACCESS SELF.SEED, AS 
    THAT WILL BE THE SAME FOR ALL SWEEPS OVER THE SAME NUMBER OF OBSTACLES:
    THE SEED MANAGER TAKES THE NUMBER OF OBSTACLES INTO ACCOUNT AND GENERATES A NEW UNIQUE SEED
    ALSO WHEN SAMPLING START AND GOAL LOCATIONS MAKE SURE TO USE THE SEED THAT HAS GONE THROUGH 
    THE SEED MANAGER, AS OTHERWISE THEY WILL BE DUPLICATED A LOT

    Args:
        SingleWorldBase (_type_): _description_
    """
    def __init__(self, 
                    seed: int, 
                    start_goal_location_type: str, 
                    start_goal_location_type_params: dict,
                    start_goal_margin_to_obstacles: float,
                    obstacle_groups: List[Dict[str, Any]],
                    **kwargs):
        super().__init__(**kwargs)
        # The base class makes sure that upstream sweep nodes override the variables you set below. 
        # Make sure that any type of calculations that are done here are done later. 
        # E.g. the seed gets converted to a different seed later on as a function of the number of obstacles. 
        # DO NOT DO THAT HERE. As the seed is swept over, and the pre-converted seed will be used instead if you do that.

        self.start_goal_location_type = start_goal_location_type
        self.start_goal_location_type_params = start_goal_location_type_params

        self.start_goal_margin_to_obstacles = start_goal_margin_to_obstacles
        self.obstacle_groups = obstacle_groups

        self.seed = seed
    
    def get_world_constructor(self) -> WorldConstructor:
        # We resolve the actual seed as a function of the number of obstacles and experiment type
        num_obstacles = ProbabilisticWorldgen.get_number_of_obstacles_from_groups(self.obstacle_groups)
        seed = WorldgenSeedManager.get_seed(self.experiment_type, num_obstacles, self.seed)

        start, goal = self._get_start_goal_locations(self.start_goal_location_type, self.start_goal_location_type_params, seed, self.start_goal_margin_to_obstacles)
        return ProbabilisticWorldConstructor(
            goal=goal, 
            start=start,
            seed=seed, 
            start_goal_margin_to_obstacles=self.start_goal_margin_to_obstacles,
            obstacle_groups=self.obstacle_groups,
            **self.get_world_information_params()
        )

    def _get_start_goal_locations(self, start_goal_location_type, start_goal_location_type_params, seed, margin) -> Tuple[np.ndarray, np.ndarray]:
        vs = self.voxel_size
        world_limits = (self.world_limits[0] + vs + margin, self.world_limits[1] - vs - margin)
        if start_goal_location_type == "random_with_mindist":
            return next(StartGoalMinDistSampler(seed).sample(start_goal_location_type_params["min_dist"], world_limits))
        elif start_goal_location_type == "single_manual":
            # In case of e.g. sampling points in the world, the start location doesn't matter, so we just put it at the goal location
            return start_goal_location_type_params["location"], start_goal_location_type_params["location"]  
        elif start_goal_location_type == "manual":
            return start_goal_location_type_params["start"], start_goal_location_type_params["goal"]
        elif start_goal_location_type == "single_random":
            goal = next(UniformWorldLimitsSampler(seed).sample(goal=None, limits=world_limits))
            return goal, goal
        else: 
            raise ValueError(f"Unknown start_goal_location_type: {start_goal_location_type}")