from __future__ import annotations

import abc
import copy
import functools
import os
import time
from typing import Callable, List, Tuple
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_base import SamplerBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampling_function import RolloutSamplingStatistics, SamplingFunction
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor
from rmp_dl.planner.planner import PlannerRmp
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, PlannerParameters, RayObserverParameters, RaycastingCudaPolicyParameters, TargetPolicyParameters
from rmp_dl.testing.random_world_tester import RandomWorldTester

import torch

class RolloutSampler(SamplerBase):
    def __init__(self, 
                 stride: int,
                 terminate_if_stuck: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.terminate_if_stuck = terminate_if_stuck
        self.stride = stride

    @abc.abstractmethod
    def _get_planner_constructor(self) -> Callable[[], PlannerRmp]: ...
        
    class RolloutSamplingFunction(SamplingFunction):
        def __init__(self, rollout_sampler: RolloutSampler):
            self.planner_constructor = rollout_sampler._get_planner_constructor()
            self.stride = rollout_sampler.stride
            self.terminate_if_stuck = rollout_sampler.terminate_if_stuck

        @staticmethod
        def _valid_position(pos, world_limits):
            return ((pos > world_limits[0]) & (pos < world_limits[1])).all()

        def setup(self) -> None:
            self.planner = self.planner_constructor()

        def __call__(self, world_constructor: WorldConstructor) -> Tuple[List[dict], RolloutSamplingStatistics]:
            world_limits = world_constructor.get_world_limits()
            vs = world_constructor.get_voxel_size()
            world_limits = (world_limits[0] + vs, world_limits[1] - vs)

            worldgen = world_constructor()
            if all(worldgen.get_start() == worldgen.get_goal()):
                raise RuntimeError("Start and goal location are identical, while using a rollout sampler. Make sure to use a different sampler or change the world generation parameters."
                                   "Specifically, make sure that the start and goal locations are not identical, for probabilistic worlds this means changing start_goal_location_type")
            

            esdf = worldgen.get_esdf() if self.planner.requires_esdf else None
            geodesic = worldgen.get_distancefield() if self.planner.requires_geodesic else None
            self.planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf=esdf, geodesic=geodesic)

            observations = []
            steps = 1
            t = time.time()
            while True:
                # For the first rollout we do a single step, so that we can get the initial position
                observation, terminated = self.planner.step(steps=steps, terminate_if_stuck=self.terminate_if_stuck)
                steps = self.stride

                if terminated:  # Termination can have some unstable data, so we exit immediately
                    break
                pos = self.planner.get_pos().copy()

                if not self._valid_position(pos, world_limits):
                    break

                observation.update({
                    "info" : {
                        "start": worldgen.get_start().copy(),
                        "goal": worldgen.get_goal().copy(),
                    }})

                observations.append(observation)

            total_time = time.time() - t
            df = RandomWorldTester.get_df(self.planner, worldgen, total_time, world_constructor.get_num_obstacles(), world_constructor.get_seed())

            return observations, RolloutSamplingStatistics(df), ""

    def _get_sampling_function(self) -> SamplingFunction:
        return self.RolloutSamplingFunction(self)
    
class LearnedRolloutSampler(RolloutSampler):
    def __init__(self, temporary_storage_path, config_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporary_storage_path = temporary_storage_path + "/temp_models_for_rollout" + f"/{os.getpid()}"
        if MpUtil.is_main_process():
            # Will be reused for multiple learned rollout nodes, so exist_ok is true
            # We have barriers before moving to other rollout nodes, so overlap is not an issue
            os.makedirs(self.temporary_storage_path, exist_ok=True) 
        self.config_path = config_path
        
    def _get_planner_constructor(self) -> Callable[[], PlannerRmp]:
        if self.model_args is None or self.model_state_dict is None:
            raise RuntimeError("Model args and state dict not set in LearnedRolloutSampler. See DataPipelineMpWrapper on how to set this.")
        model_path =  self.temporary_storage_path + f"/temp_model.pt"
        model_args, state_dict = self.model_args, self.model_state_dict
        if MpUtil.is_main_process():
            torch.save(state_dict, model_path)
        MpUtil.barrier() 
        model_args = copy.deepcopy(model_args)

        raycasting_cuda_parameters = RaycastingCudaPolicyParameters.from_yaml_general_config(self.config_path)
        learned_policy_rmp_params = LearnedPolicyRmpParameters.from_yaml_general_config(self.config_path)
        planner_params = PlannerParameters.from_yaml_general_config(self.config_path)
        ray_observer_params = RayObserverParameters.from_yaml_general_config(self.config_path)

        return functools.partial(PlannerFactory.learned_labeled_from_path, model_path, model_args, 
                                 raycasting_cuda_parameters, learned_policy_rmp_params, ray_observer_params, planner_params) # type: ignore

class ExpertRolloutSampler(RolloutSampler):
    def __init__(self, config_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = config_path
        
    def _get_planner_constructor(self) -> Callable[[], PlannerRmp]:
        target_policy_params = TargetPolicyParameters.from_yaml_general_config(self.config_path)
        raycasting_cuda_parameters = RaycastingCudaPolicyParameters.from_yaml_general_config(self.config_path, expert=True)
        planner_params = PlannerParameters.from_yaml_general_config(self.config_path)
        ray_observer_params = RayObserverParameters.from_yaml_general_config(self.config_path)

        return functools.partial(PlannerFactory.expert_labeled, target_policy_params, raycasting_cuda_parameters, ray_observer_params, planner_params) # type: ignore