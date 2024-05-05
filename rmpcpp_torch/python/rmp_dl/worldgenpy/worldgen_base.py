import abc
from typing import Optional, Tuple
import numpy as np
from rmp_dl.util.voxel_conversions import VoxelConversions
from rmp_dl.planner.planner_params import WorldgenSettings
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen

from worldgenBindings import WorldGenCPP
from nvbloxLayerBindings import TsdfLayer, EsdfLayer
from nvbloxConversionsBindings import esdf_distance_to_vector, tsdf_distance_to_vector

import rmp_dl.util.io as rmp_io

class WorldgenBase(abc.ABC):
    def __init__(self, settings: WorldgenSettings):
        self._settings = settings
        self._start = None
        self._goal = None
    
    def get_start(self) -> np.ndarray:
        if self._start is None:
            raise RuntimeError("Start position not set")
        return self._start
    
    def get_goal(self) -> np.ndarray:
        if self._goal is None:
            raise RuntimeError("Goal position not set")
        return self._goal

    def set_start(self, startpos: np.ndarray):
        self._start = startpos

    def set_goal(self, goal: np.ndarray):
        self._goal = goal

    def get_settings(self):
        return self._settings

    def export_to_ply(self, save_dir: Optional[str]=None) -> str:
        path: str = save_dir if save_dir is not None else rmp_io.resolve_directory(f"data/temp/ply/temp.ply")
        WorldGenCPP.exportToPly(path, self.get_tsdf())

        return path

    def get_info(self) -> dict:
        """Returns an information dictionary about the world
        """
        return {}  # Up to derivates to implement this

    @abc.abstractmethod
    def get_tsdf(self) -> TsdfLayer: ...
    
    @abc.abstractmethod
    def get_esdf(self) -> EsdfLayer: ...
    
    @abc.abstractmethod
    def get_density(self) -> float: ...
    
    def get_distancefield(self, df: np.ndarray=None, inflation=0.0) -> DistanceFieldGen:
        """Get the distance field for the world

        Args:
            df (_type_, optional): Can be used if we have a cached distancefield. Defaults to None.
        """
        esdf_as_array = WorldgenBase.get_sdf_as_ndarray_static(self.get_esdf(), self.get_world_limits(), self.get_voxel_size())
        return DistanceFieldGen(esdf_as_array, self.get_goal(), self.get_world_limits(), self.get_voxel_size(), df=df, inflation=inflation)
    
    @staticmethod
    def clean_sdf(sdf, world_limits, voxel_size):
        # because nvblox works with 8x8x8 blocks, the beginning and last part of the esdf may be useless
        # so we remove those parts. 
        shape = sdf.shape

        low = np.rint(world_limits[0] / voxel_size) % 8
        high = 8 - np.rint(world_limits[1] / voxel_size) % 8
        low = np.rint(low).astype(int); high = np.rint(high).astype(int)
        sdf = sdf[low[0]:-high[0], low[1]:-high[1], low[2]:-high[2]]

        grid_size = np.rint((world_limits[1] - world_limits[0]) \
                     / voxel_size).astype(int)

        # assert (grid_size + low + high == shape).all()
        # assert (sdf.shape == grid_size).all()
        return sdf

    def get_tsdf_as_ndarray(self) -> np.ndarray:
        return self.get_sdf_as_ndarray_static(self.get_tsdf(), self._settings.world_limits, self._settings.voxel_size)

    @staticmethod
    def get_sdf_as_ndarray_static(sdf: TsdfLayer, world_limits: Tuple[np.ndarray, np.ndarray], voxel_size: float) -> np.ndarray:
        shape, vector = tsdf_distance_to_vector(sdf)
        sdf = np.array(vector).reshape(shape)
        
        # In the case where we have world limits, we can remove the padding from the sdf
        if world_limits is not None:
            sdf = WorldgenBase.clean_sdf(sdf, world_limits, voxel_size)

        return sdf
        
    def get_esdf_as_ndarray(self) -> np.ndarray:    
        # Note that the underlying datastructure of the esdf is still a tsdf, just with a very high truncation distance. 
        return self.get_sdf_as_ndarray_static(self.get_esdf(), self._settings.world_limits, self._settings.voxel_size)

    def get_voxel_size(self) -> float:
        return self._settings.voxel_size
    
    def get_world_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._settings.world_limits