from typing import List, Optional
import numpy as np
import pandas as pd
from rmp_dl.learning.data.pipeline.nodes.samplers.sampling_function import SamplingStatistics
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor

class SamplerOutputBase:
    def __init__(self, observations: dict, 
                 world_constructor: WorldConstructor, 
                 stats: Optional[SamplingStatistics] = None):
        """Observations is a dictionary of observations in a world. We use numpy arrays for the observations, where the first 
        dimension is the number of samples in that world (for sequential data it corresponds to timesteps),
        and the second dimension is the dimensionality of the observation.
        It used to be that observations were a list of dictionaries, and that the numpy arrays would be 1d. 
        However, this is much slower to deal with and pass around, e.g. reading from disk is ~3 times slower. 
        And we want to be able to serialize these outputs quickly in case we are training with a lot of data, and 
        we use disk caching. 
        Note that the first dimension of all the numpy arrays within the observation dictionaries are equal to each other.
        Example observation could be (example sequence length 100):
        {
            "state": {
                "pos": np.ndarray[100, 3]
                "vel": np.ndarray[100, 3]
                }
            "rays": {"rays": np.ndarray[100, 1024]} 
            "expert_policy": {"geodesic": np.ndarray[100, 3]}  
            "info": {"goal": np.ndarray[100, 3], "start": np.ndarray[100, 3]} 
        }
        The goal and start locations are kind of a waste of space, as they are the same for all samples in the batch.
        It is easier to deal with though 
        """
        self._observations = observations
        self._world_constructor = world_constructor
        self._stats = stats

    def get_observations(self) -> dict:
        return self._observations

    def get_world_constructor(self) -> WorldConstructor:
        return self._world_constructor
    
    def get_stats(self) -> Optional[SamplingStatistics]:
        return self._stats

    @staticmethod
    def length_of_first_numpy_array_in_dict(d) -> int:
        """
        We just find the first numpy array that we can find to get the length. 
        The arrays should all be of equal length anyway
        """
        if isinstance(d, np.ndarray):
            return d.shape[0]
        elif isinstance(d, dict):
            return SamplerOutputBase.length_of_first_numpy_array_in_dict(list(d.values())[0])
        raise ValueError("Expected a dict or np.ndarray, got {}".format(type(d).__name__))

    def __len__(self):
        return self.length_of_first_numpy_array_in_dict(self._observations)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise RuntimeError("SamplerOutputBase only supports integer indexing")
        def extract(d, i):
            if isinstance(d, np.ndarray):
                return d[[i]] # We maintain the 1st dimension
            elif isinstance(d, dict):
                return {k: extract(v, i) for k, v in d.items()}
            raise ValueError("Expected a dict or np.ndarray, got {}".format(type(d).__name__))
        return extract(self._observations, index)
    