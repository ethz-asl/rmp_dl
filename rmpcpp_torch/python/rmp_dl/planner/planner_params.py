
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Tuple
import numpy as np
import rmp_dl.util.io as rmp_io

from rmpPlannerBindings import PlannerParametersCpp


"""All these dataclasses can be loaded from a general yaml config file.
If you pass path=None, note that it takes the default directory in the config folder. 
Do not use this function with path=None for any training code, as the training code
is supposed to use copied configuration files in a different folder
"""

@dataclass()
class LearnedPolicyRmpParameters:
    alpha: float
    beta: float
    c_softmax: float

    @staticmethod
    def from_yaml_general_config(base_path=None) -> LearnedPolicyRmpParameters:
        params = rmp_io.ConfigUtil.get_yaml_general_params(base_path)
        return LearnedPolicyRmpParameters(**params["policies"]["learned"])

@dataclass()
class TargetPolicyParameters:
    alpha: float 
    beta: float 
    c_softmax: float 
    metric: float 

    @staticmethod
    def from_yaml_general_config(base_path=None) -> TargetPolicyParameters:
        params = rmp_io.ConfigUtil.get_yaml_general_params(base_path)
        return TargetPolicyParameters(**params["policies"]["target"])


@dataclass()
class RaycastingCudaPolicyParameters:
    eta_rep: float 
    eta_damp: float 
    v_rep: float 
    v_damp: float 
    epsilon_damp: float 
    c_softmax_obstacle: float 
    r: float 

    metric: bool 
    N_sqrt: int 
    surface_distance_epsilon_vox: float 
    max_steps: int 
    truncation_distance_vox: float 

    metric_scale: float
    force_scale: float

    @staticmethod
    def from_yaml_general_config(base_path=None, expert=False, name="raycasting_avoidance") -> RaycastingCudaPolicyParameters:
        """Return the parameters from the yaml file

        Args:
            expert (bool, optional): If we use an expert policy or not (as we use different params in that case). 
                Defaults to False.
        """
        params = rmp_io.ConfigUtil.get_yaml_general_params(base_path)
        name = "raycasting_avoidance_geodesic" if expert else name
        return RaycastingCudaPolicyParameters(
            **params["policies"][name],
            truncation_distance_vox=params["truncation_distance_vox"]
            )


@dataclass()
class RayObserverParameters:
    N_sqrt: int 
    # This is a relative measure in voxels, so gets multiplied by voxel_size later on
    truncation_distance_vox: float 
    maximum_steps: int 
    maximum_ray_length: float 
    surface_distance_epsilon_vox: float 

    @staticmethod
    def from_yaml_general_config(base_path=None) -> RayObserverParameters:
        params = rmp_io.ConfigUtil.get_yaml_general_params(base_path)
        return RayObserverParameters(
            **params["policies"]["raycasting_dl"],
            )


@dataclass()
class PlannerParameters:
    dt: float 
    max_length: int 
    # This is a relative measure in voxels, so gets multiplied by voxel_size later on
    truncation_distance_vox: float 
    terminate_upon_goal_reached: bool 

    @staticmethod
    def from_yaml_general_config(path=None) -> PlannerParameters:
        params = rmp_io.ConfigUtil.get_yaml_general_params(path)
        params = {**params["planner"], "truncation_distance_vox": params["truncation_distance_vox"]}
        return PlannerParameters(**params)

    def to_cpp(self):
        return PlannerParametersCpp(**self.__dict__)

@dataclass()
class WorldgenSettings:
    world_limits: Tuple[np.ndarray, np.ndarray]

    voxel_size: float
    voxel_truncation_distance_vox: float

    @staticmethod
    def from_yaml_general_config(path=None) -> WorldgenSettings:
        params = rmp_io.ConfigUtil.get_yaml_general_params(path)
        return WorldgenSettings(
            world_limits=(
                np.array(params["worldgen"]["world_limits"]["min"]),
                np.array(params["worldgen"]["world_limits"]["max"])
            ),
            voxel_size=params["voxel_size"],
            voxel_truncation_distance_vox=params["truncation_distance_vox"]
            )