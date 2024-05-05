import numpy as np
from typing import Tuple

class VoxelConversions:

    def __init__(self, world_limits, voxel_size):
        if not isinstance(world_limits, tuple) and len(world_limits.shape) == 1:
            self.world_limits = (np.zeros_like(world_limits), world_limits)
        else:
            self.world_limits = (np.array(world_limits[0]), np.array(world_limits[1]))
        self.voxel_size = voxel_size
        self.dims = self.world_limits[0][:].shape[0]

    def convertPositionToVoxelIndex(self, position: np.ndarray) -> np.ndarray:
        return np.floor((position - self.world_limits[0]) / self.voxel_size)

    def convertVoxelIndexToVoxelCenter(self, index: np.ndarray) -> np.ndarray: 
        return index * self.voxel_size + self.world_limits[0]

    def distanceToVoxelCenter(self, position: np.ndarray, voxel_index: np.ndarray) -> float:
        return np.norm(position - self.convertVoxelIndexToVoxelCenter(voxel_index))