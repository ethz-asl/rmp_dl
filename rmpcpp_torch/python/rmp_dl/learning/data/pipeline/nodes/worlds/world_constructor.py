from __future__ import annotations

import abc
from typing import Any, Callable, Iterable, Optional, Tuple, final
import numpy as np

from rmp_dl.worldgenpy.worldgen_base import WorldgenBase


class WorldConstructor(abc.ABC):
    """Just a wrapper to make it clear what we are passing around. We also add a description, and 
    information about which class made this constructor, along with the arguments used. 
    EDIT2: The stuff below is somewhat outdated. After refactoring it has become a bit cleaner, and 
    we don't serialize to wandb anymore, but I may want to re-implement that in the future. 
    EDIT: This is kind of growing with more information now. 
    The data in this class is very non-rigid, e.g. lots of dicts with no type information.
    This is because this class gets serialized when saving to wandb, so we don't want rigid types as this makes deserialization difficult.
    Also using dicts is just quite flexible in general. 
    Ideally, I write conversions for this class to and from more integral types for use in serialization, 
    and then put more rigid types in and around everything that uses this class,
    but for now I'll leave this as is. 
    Important to keep in mind for this class is that _single_world_params are the parameters used to create the single world that created this constructor.
    Look at the _set_params method in SingleWorldBase for parameters that every single world has, so we can write getter methods for them here. 
    """
    def __init__(self, world_limits, voxel_size, voxel_truncation_distance_vox):
        self.world_limits = world_limits
        self.voxel_size = voxel_size
        self.voxel_truncation_distance_vox = voxel_truncation_distance_vox
        # Used to indicate whether we should inflate this world when calculating the geodesic field. 
        self._inflation = 0.0

    @abc.abstractmethod
    def __call__(self, *, continued_generation) -> WorldgenBase: 
        """Continued generation is used to continue the generation of a world, e.g. such that successive
        calls result in different worlds. Only works for non-static worlds (e.g. random obstacle world).
        """
        ...
        
    @abc.abstractmethod
    def __str__(self) -> str: ...

    @property
    def inflation(self):
        return self._inflation
    
    @inflation.setter
    def inflation(self, value):
        self._inflation = value

    def get_world_limits(self):
        return self.world_limits

    def get_voxel_size(self):
        return self.voxel_size
    
    def get_voxel_truncation_distance_vox(self):
        return self.get_voxel_truncation_distance_vox

    def get_seed(self):
        return getattr(self, "seed", -1)
    
    def get_num_obstacles(self):
        return getattr(self, "num_obstacles", -1)

    @abc.abstractmethod
    def _get_world_equality_hashable_params(self) -> Tuple:
        """We use this to compare whether two worlds would be identical (i.e. same voxels if created). 
        So subclasses should add the critical parameters that would affect the voxels into nested tuples and return it. 
        For example, voxel truncation distance actually does not affect the voxels (just the TSDF/ESDF), so it should not be included.
        Or in case of a predefined world with a certain start and goal location, these locations do not affect the world itself. 
        In the random obstacle world however, they DO affect the world, as the randomized obstacles are not put at the same location as the start and goal, 
        therefore the world will be different if the start and goal locations are different.
        This equality is used to determine whether 2 trajectories should be plotted within the same world when logging open3d images in the datapipeline, 
        see the class open3d rendering in the datapipeline.
        Using this method is much faster than actually generating the worlds and comparing the voxels, 
        and in the end we only have to generate a single world to plot all the trajectories within the same world. 
        
        Subclasses should call this method to get the base class parameters, and then concatenate their own parameters
        """
        hashable_world_limits = tuple(map(tuple, self.get_world_limits()))
        return (self.get_voxel_size(), hashable_world_limits)

    @staticmethod
    def to_hashable(d):
        if isinstance(d, dict):
            return tuple((k, WorldConstructor.to_hashable(v)) for k, v in sorted(d.items()))
        if isinstance(d, list):
            return tuple(WorldConstructor.to_hashable(v) for v in d)
        if isinstance(d, np.ndarray):
            return tuple(d.flatten())
        return d

    @final
    def __eq__(self, other: WorldConstructor):
        return self._get_world_equality_hashable_params() == other._get_world_equality_hashable_params()

    @final
    def __hash__(self):
        """Similarly to above, we use this to hash the worlds, only hashing the parameters that would affect the voxels."""
        return hash(self._get_world_equality_hashable_params())
    

    def _get_open3d_camera_settings(self):
        wl_min, wl_max = self.world_limits
        min_x, min_y, min_z = wl_min
        max_x, max_y, max_z = wl_max

        sx, sy, sz = max_x - min_x, max_y - min_y, max_z - min_z


        look_at = np.array([
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2,
        ])

        # This is relative to look at 
        position = np.array([
            sx * 1.8, 
            sy * 1.2,
            - sz * 0.3
        ])
        
        up = np.array([0, 1, 0])

        return position, look_at, up