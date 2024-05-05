import numpy as np
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase, WorldgenSettings
from nvbloxSceneBindings import NvbloxSphere, NvbloxCube, NvbloxScene, \
    NvbloxPrimitive, generate_tsdf_from_scene

from worldgenBindings import WorldGenCPP
from nvbloxLayerBindings import EsdfLayer, TsdfLayer, MemoryType


class CustomWorldgen(WorldgenBase):
    MEMORY_TYPE = MemoryType.kUnified

    def __init__(self, 
                 settings: WorldgenSettings = WorldgenSettings.from_yaml_general_config(),
                 ):
        super().__init__(settings)
        self._scene = NvbloxScene()
        self._scene.set_aabb(self._settings.world_limits[0], self._settings.world_limits[1])
        self._tsdf = None
        self._esdf = None

    def _reset_distance_fields(self):
        self._tsdf = None
        self._esdf = None

    def add_primitive(self, primitive: NvbloxPrimitive):
        self._scene.add_primitive(primitive)

    def add_cube(self, loc: np.ndarray, size: np.ndarray):
        self._reset_distance_fields()
        cube = NvbloxCube(loc, size)
        self._scene.add_primitive(cube)

    def add_sphere(self, loc: np.ndarray, radius: float):
        self._reset_distance_fields()
        sphere = NvbloxSphere(loc, radius)
        self._scene.add_primitive(sphere)

    def add_bounds_automatic(self):
        ([xmin, ymin, zmin], [xmax, ymax, zmax]) = self._settings.world_limits
        vxs = self._settings.voxel_size
        self._scene.add_ground_level(zmin + vxs)
        self._scene.add_ceiling(zmax - vxs)
        self._scene.add_plane_boundaries(xmin + vxs, xmax - vxs, ymin + vxs, ymax - vxs)

    def get_esdf(self) -> TsdfLayer:
        """Get the esdf. Note that we use a tsdf layer with truncation distance set to a large value. 
        The built in nvblox EsdfLayer is less accurate than the TsdfLayer, and it uses voxel distance 
        instead of actual distance. 

        Returns:
            TsdfLayer: TsdfLayer with a very high truncation distance. 
        """
        TRUNC_DIST = 100.0
        if self._esdf is None:
            self._esdf = TsdfLayer(self._settings.voxel_size, CustomWorldgen.MEMORY_TYPE)
            generate_tsdf_from_scene(self._scene, 
                                     TRUNC_DIST,
                                     self._esdf)
        return self._esdf

    def get_tsdf(self) -> TsdfLayer:
        if self._tsdf is None:
            self._tsdf = TsdfLayer(self._settings.voxel_size, CustomWorldgen.MEMORY_TYPE)
            generate_tsdf_from_scene(self._scene, 
                                     self._settings.voxel_truncation_distance_vox * self._settings.voxel_size, 
                                     self._tsdf)
        return self._tsdf

    def get_density(self) -> float:
        #TODO: Refactor this to the baseclass (look at worldgen random cpp)
        return 0.0