
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import State
from rmp_dl.util.halton_sequence import HaltonUtils
import torch


class PartialRayObserver(Observer):
    def __init__(self, 
                 num_rays: int,
                 target: np.ndarray,
                 sensor_direction, 
                 fov: float,
                 ray_observation_getter: Observer):
        super().__init__()
        # These are the unit vector endpoints of the halton sequence rays. Precompute them     
        points = HaltonUtils.get_ray_endpoints_from_halton_distances(np.ones(num_rays))
        self.endpoints = torch.from_numpy(points).to(torch.float32).to(torch.device("cuda")) # (N, 3
        self.ray_observation_getter = ray_observation_getter

        self.target = target

        self.sensor_direction = sensor_direction
        self.fov = fov

    def _get_observation(self, state: State) -> torch.Tensor:
        rays = self.ray_observation_getter(state)
    
        forward_direction = self._get_forward_direction(state)
        rotation_matrix = self._get_rotation_matrix(forward_direction)

        # Rotate the sensor direction to the actual sensor direction based on the forward direction of the agent
        sensor_direction = rotation_matrix @ self.sensor_direction

        rays = self._resolve_partial_observation(rays, sensor_direction)
        return rays

    def _resolve_partial_observation(self, rays: torch.Tensor, sensor_direction: np.array):
        sensor_direction = torch.from_numpy(sensor_direction).to(torch.float32).to(torch.device("cuda"))
        observation = torch.zeros_like(rays, dtype=torch.float32, device="cuda")

        # First, we always choose the closest ray to the sensor direction
        closest_ray_idx = torch.argmax(self.endpoints @ sensor_direction)
        observation[closest_ray_idx] = rays[closest_ray_idx]
        
        # Next, we set the rays that are within the fov of the sensor direction
        dot_products = self.endpoints @ sensor_direction
        fov_mask = dot_products > np.cos(self.fov / 2)
        observation[fov_mask] = rays[fov_mask]

        return observation

    def _get_forward_direction(self, state: State):
        vel: np.array = state.vel
        if np.linalg.norm(vel) < 1e-7:
            rel_pos = self.target - state.pos
            return rel_pos / np.linalg.norm(rel_pos)

        return vel / np.linalg.norm(vel)
    
    @staticmethod
    def _get_rotation_matrix(forward_direction: np.array):
        # We assume that the sensor_direction is relative to a fixed frame that has forward facing 
        # direction [1, 0, 0]. In this function, we calculate the rotation matrix that 
        # rotates the fixed frame forward direction to the actual forward direction of the agent.
        # Using this rotation matrix we can rotate the sensor direction to the actual sensor direction based 
        # on the forward direction of the agent.
        # First we do yaw (rotation around z-axis) and then pitch (rotation around y-axis). 
        # We assume that the roll of the agent is always 0.
        forward_direction = forward_direction / np.linalg.norm(forward_direction)
        
        yaw_angle = np.arctan2(forward_direction[1], forward_direction[0])
        R_yaw = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1]
        ])

        # Undo the yaw rotation of the forward direction
        forward_direction_adj = R_yaw.T @ forward_direction

        # Pitch rotation (around Y-axis)
        pitch_angle = -np.arctan2(forward_direction_adj[2], forward_direction_adj[0])
        R_pitch = np.array([
            [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
            [0, 1, 0],
            [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
        ])

        return R_yaw @ R_pitch 



        # forward_direction = forward_direction / np.linalg.norm(forward_direction)
        # fixed_forward = np.array([1, 0, 0])

        #  # Calculate the rotation axis (perpendicular to both forward directions)
        # v = np.cross(fixed_forward, forward_direction)

        # if np.linalg.norm(v) < 1e-7:
        #     if np.dot(fixed_forward, forward_direction) > 0:
        #         return np.eye(3)
        #     else:
        #         return np.array([
        #             [-1, 0, 0],
        #             [0, 1, 0],
        #             [0, 0, 1]
        #         ])

        # # Rotation angles
        # c = np.dot(fixed_forward, forward_direction)
        # s = np.linalg.norm(v)

        # v = v / np.linalg.norm(v)

        # v1, v2, v3 = v

        # # Rodrigues' rotation formula
        # K = np.array([
        #     [0, -v3, v2],
        #     [v3, 0, -v1],
        #     [-v2, v1, 0]
        # ])

        # rotation_matrix = np.eye(3) + K * s + K @ K * (1 - c)

        # return rotation_matrix




        
if __name__ == "__main__":
    direction = np.array([1, 1, 1])

    rot = PartialRayObserver._get_rotation_matrix(direction)