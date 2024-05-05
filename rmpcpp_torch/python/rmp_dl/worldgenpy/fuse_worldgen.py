import os
from typing import Optional
import numpy as np

from rmp_dl.planner.planner_params import WorldgenSettings
from nvbloxLayerBindings import TsdfLayer
from fuse3DMatchBindings import Fuse3DMatchWorldgen
from nvbloxSerializerBindings import NvbloxSerializer
from nvbloxConversionsBindings import get_aabb_of_observed_voxels, get_min_indices, get_max_indices, tsdf_to_esdf_with_tsdf_voxeltype

import rmp_dl.util.io as rmp_io

from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

class Fuser:
    def __init__(self, base_path: str, timing_path: str, map_output_path: str, mesh_output_path: str, settings: WorldgenSettings, trunc_dist_vox: float=4.0):
        self.fuser = Fuse3DMatchWorldgen(base_path, timing_path, map_output_path, mesh_output_path)
        self.fuser.set_voxel_size(settings.voxel_size)
        # With the fuser we just generate a tsdf with a high truncation distance. 
        # For chomp this is useful, as you then have gradients everywhere
        # For the rmp planner it makes it maybe slightly faster as it can step faster with the rays. 
        # There is not much point actually to using the tsdf for the rmp planner other than showing the 
        # ease of use for any world representation. 
        self.fuser.set_truncation_distance_vox(trunc_dist_vox)

    def run(self):
        self.fuser.run()

class FuseWorldGen(WorldgenBase):
    def __init__(self, base_path: str, output_name: str="trunc_4.0", trunc_dist_vox: float=4.0):
        super().__init__(WorldgenSettings.from_yaml_general_config())
        self.set_goal(np.array([0.0, 0.0, 0.0]))
        self.set_start(np.array([0.0, 0.0, 0.0]))
        self._create_paths(base_path, output_name)

        print(f"Trying to load tsdf from {self.map_output_path}")
        self.tsdf = NvbloxSerializer.load_tsdf_from_file(self.map_output_path)
        if self.tsdf == None:
            print("No map file found, generating a new one")
            self.fuser = Fuser(base_path, self.timing_path, self.map_output_path, self.mesh_output_path, self._settings, trunc_dist_vox)
            self.fuser.run()
            self.tsdf = NvbloxSerializer.load_tsdf_from_file(self.map_output_path)

            if self.tsdf == None:
                raise Exception("No map file found, and generation failed")
        
        print(f"Voxel Size: {self.tsdf.get_voxel_size()}")
        self.esdf = TsdfLayer(self.tsdf.get_voxel_size(), self.tsdf.memory_type())
        tsdf_to_esdf_with_tsdf_voxeltype(self.tsdf, self.esdf)

        # world_limits = get_aabb_of_observed_voxels(self.tsdf)
        min_indices = np.array(get_min_indices(self.tsdf))
        max_indices = np.array(get_max_indices(self.tsdf))

        world_limits = (min_indices * self.tsdf.get_voxel_size(), (max_indices + 1) * self.tsdf.get_voxel_size())
        aabb = get_aabb_of_observed_voxels(self.tsdf)

        self._settings.world_limits = world_limits
        self._settings.voxel_size = self.tsdf.get_voxel_size()

    def _create_paths(self, base_path: str, output_name):
        base_path += "/" + output_name
        timing_path = base_path + "/timing"
        map_path = base_path + "/map"
        mesh_path = base_path + "/mesh"
        os.makedirs(timing_path, exist_ok=True)
        os.makedirs(map_path, exist_ok=True)
        os.makedirs(mesh_path, exist_ok=True)

        self.timing_path = timing_path + "/timing.txt"
        self.map_output_path = map_path + "/map.nvb"
        self.mesh_output_path = mesh_path + "/mesh.ply"

    def export_to_ply(self, save_dir: Optional[str]=None) -> str:
        if save_dir is not None:
            raise Exception("Save dir not supported for fuse worldgen")
        return self.mesh_output_path

    def get_tsdf(self):
        return self.tsdf

    def get_esdf(self):
        return self.esdf

    def get_density(self) -> float:
        # TODO: Implement this
        return 0.0


if __name__ == "__main__":
    from rmp_dl.vis3d.planner3dvis import Planner3dVis
    base_path = rmp_io.get_temp_data_dir() + "/3dmatch/sun3d-mit_32_d507-d507_2"

    worldgen = FuseWorldGen(base_path)

    planner3dvis = Planner3dVis(worldgen, distancefield=False)
    planner3dvis.go()