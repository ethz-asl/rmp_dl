

import abc
from typing import List, Tuple
from attr import dataclass

import numpy as np
from rmp_dl.util.halton_sequence import HaltonUtils
from rmp_dl.vis3d.cinema.actor import ActorBase
from rmp_dl.vis3d.cinema.actors.dummy_actor import DummyActor
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
import scipy
from sklearn.decomposition import PCA

from scipy.spatial.transform import Rotation as R


class CameraBase(abc.ABC):
    @dataclass()
    class View:
        pos: np.ndarray
        lookat: np.ndarray
        up: np.ndarray
        zoom: float = 0.1

        def __iter__(self):
            return iter((self.pos, self.lookat, self.up, self.zoom))

    @abc.abstractmethod
    def get_view(self) -> View: ...

class CameraRotating(CameraBase):
    def __init__(self, center: np.ndarray, radius: float, speed: float, up: np.ndarray):
        self.center = center
        self.radius = radius
        self.speed = speed
        self.up = up / np.linalg.norm(up)

        self.angle = 0.0

    def get_view(self) -> CameraBase.View:

        rot = R.from_rotvec(self.up * self.angle)
        pos = self.center + np.array([self.radius, 0.0, 0.0]) @ rot.as_matrix()
        print(pos)

        self.angle += self.speed

        return CameraBase.View(pos, self.center, self.up, 1)
        
    def get_actor(self) -> ActorBase:
        # We return a dummy actor that makes sure that we keep going for 1 rotation
        # Calculate number of steps for 1 rotation
        steps = int(2 * np.pi / self.speed) - 1
        return DummyActor(steps)

class CameraFixed(CameraBase):
    def __init__(self, pos: np.ndarray, lookat: np.ndarray, up: np.ndarray, zoom: float = 0.1):
        self.pos = np.array(pos)
        self.lookat = np.array(lookat)
        self.up = np.array(up)
        self.zoom = zoom

    def get_view(self) -> CameraBase.View:
        return CameraBase.View(self.pos, self.lookat, self.up, self.zoom)


class FilterPositionCamera(CameraBase):
    def __init__(self, *, height: float = 0.1, filter_cutoff: float = 0.02):
        self.height = height

        initial_relative_camera_position = np.array([-0.5, 2.0, -0.5]) + np.array([0.0, self.height, 0.0])

        self.filter = scipy.signal.butter(2, filter_cutoff, 'low', analog=False, output='sos')

        # We add current positions of our camera and heavily filter those to get a delayed effect
        self.camera_positions = [initial_relative_camera_position] * 100
        self.observation = {}

        self.up = np.array([0.0, 1.0, 0.0])

        self.lookat = None

    def set_lookat_and_camera_pos(self, lookat, camera_position: np.ndarray):
        self.lookat = lookat
        
        # We add a bit of height to the camera
        # DONT USE += HERE, IT WILL CHANGE THE ORIGINAL ARRAY
        camera_position = camera_position + np.array([0.0, self.height, 0.0])
        self.camera_positions.append(camera_position.copy())

    def get_view(self) -> CameraBase.View:
        if self.lookat is None:
            raise RuntimeError("Set lookat first using the set_lookat_and_camera_pos method. (Using a callback is recommended)")
        # We filter the positions to get a delayed effect
        filtered_camera_pos = scipy.signal.sosfilt(self.filter, self.camera_positions, axis=0)[-1]
        
        # Pos of camera is relative to lookkat!
        filtered_camera_pos = filtered_camera_pos - self.lookat

        return CameraBase.View(filtered_camera_pos, self.lookat, self.up)
    
    @staticmethod
    def get_PCA_axis(observation):
        rays = observation["learned_policy"]["output_ray_predictions"].copy()
        rays = np.exp(rays)
        rays = rays / np.sum(rays)

        # Normalize between 0 and 1
        rays = (rays - np.min(rays)) / (np.max(rays) - np.min(rays))

        endpoints = HaltonUtils.get_ray_endpoints_from_halton_distances(rays)

        # Get PCA axis
        pca = PCA()
        pca.fit(endpoints)

        return pca.components_[-1]


