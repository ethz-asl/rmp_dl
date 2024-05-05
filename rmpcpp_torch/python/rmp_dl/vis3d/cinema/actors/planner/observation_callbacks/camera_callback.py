
import abc
from typing import Dict, Optional
import numpy as np
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import GeneralCallback
from rmp_dl.vis3d.cinema.camera import FilterPositionCamera


class FollowingCameraCallback(GeneralCallback):
    def __init__(self, filter_position_camera: FilterPositionCamera):
        self.filter_position_camera = filter_position_camera

    def __call__(self, observation: Optional[Dict]) -> None:
        if observation is None:
            raise IndexError("Observation is None")
        
        lookat = observation["state"]["pos"]
        camera_position = self._get_camera_position_from_observation(observation)

        self.filter_position_camera.set_lookat_and_camera_pos(lookat, camera_position)

    @abc.abstractmethod
    def _get_camera_position_from_observation(self, observation: Dict) -> np.ndarray: ...

class PositionFollowingCameraCallback(FollowingCameraCallback):
    def _get_camera_position_from_observation(self, observation: Dict) -> np.ndarray:
        # We just return the position of the agent. As the filter position camera heavily filters this, 
        # it becomes delayed and it looks like we are behind the subject
        return observation["state"]["pos"]


class PcaFollowingCameraCallback(FollowingCameraCallback):
    def _get_camera_position_from_observation(self, observation: Dict) -> np.ndarray:
        return observation["state"]["pos"] - 4.5 * FilterPositionCamera.get_PCA_axis(observation)



    