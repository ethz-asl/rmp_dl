from __future__ import annotations
import abc
from typing import Any, Iterable, List, Tuple, Union, final
import numpy as np
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor

class SingleWorldBase(PipelineObjectBase):
    def __init__(self, 
                 world_limits: Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]], 
                voxel_size: float, voxel_truncation_distance_vox: float, 
                *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.world_limits = (np.array(world_limits[0]), np.array(world_limits[1]))
        self.voxel_size = voxel_size
        self.voxel_truncation_distance_vox = voxel_truncation_distance_vox

    @final
    def get_world_information_params(self) -> dict:
        return {
            "world_limits": self.world_limits,
            "voxel_size": self.voxel_size,
            "voxel_truncation_distance_vox": self.voxel_truncation_distance_vox,
        }

    @abc.abstractmethod
    def get_world_constructor(self) -> WorldConstructor: ...

    def __iter__(self):
        # If we have no input, we just return the world constructor
        if not self._get_inputs():
            yield self.get_world_constructor()

        # If we have an input, we iterate over it and update our variables
        for input_data in self._get_input():
            if not isinstance(input_data, list) or not all(isinstance(x, Tuple) for x in input_data) or \
                not all(isinstance(x[0], str) for x in input_data):
                raise ValueError("Input data should be a list of tuples, where the first element is a string and the second element is the value.")
            
            self._update_variables(input_data)
            yield self.get_world_constructor()

    def _update_variables(self, input_data):
        for path, value in input_data:
            self.replace_variable(path, self.__dict__, value)


    @staticmethod
    def replace_variable(path, dic, value):
        """Replaces a variable in the nested dictionary (can have lists as well) determined by path with the value
        """
        # Split into parts and loop over them to get the correct attribute
        parts = path.split("/")
        for part in parts[:-1]:
            if str.isnumeric(part): part = int(part)

            try:
                dic = dic[part]
            except Exception as _:
                raise AttributeError(f"Unexpected attribute: {path}")
        
        # We now set the final value of the dict
        p = parts[-1]
        if str.isnumeric(parts[-1]): p = int(parts[-1])

        try: 
            dic[p] = value
        except Exception as _:
            raise AttributeError(f"Unexpected attribute: {path}")
