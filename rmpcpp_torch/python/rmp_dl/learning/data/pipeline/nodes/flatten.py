
from typing import List, Union
import numpy as np
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase

class Flatten(PipelineObjectBase):
    def __init__(self, sequenced: bool=False, **kwargs):
        """The flattened node transforms a list of SamplerOutputBase into 2 possible views of the data. 
        SamplerOutputBase contains world information about how the data is samped, and the actual data.
        The actual data is a (nested) dictionary of np.ndarrays. The first dimension of these arrays is the
        number of samples, and the second dimension is the dimensionality of the data. In case of sequential data, 
        the first dimension is the sequence length. But we also have e.g. densely sample data, in which the 
        first dimension does not really carry extra meaning. 
        The flattened node discards the world information, and only returns a (nested) dictionary of np.ndarrays.
        E.g. say we have 2 outputs from a sampler, the first with 100 samples, the 2nd with 200 samples. 
        The first output has a dictionary with:
        {
            "stuff": np.ndarray[100, 3]
        }
        The second output has a dictionary with:
        {
            "stuff": np.ndarray[200, 3]
        }
        
        The 2 modes of operation (sequenced and non-sequenced) will then result in:
        - If sequenced is True, the sequence length is maintained, and the flattened node simply indexes the above dictionaries, 
        and has as a length 2 (it does strip the world information, so it outputs a dictionary, and not a SamplerOutputBase).
        - If sequenced is False, the sequence length is flattened. So the flattened node will return a length of 300, and returns nodes with:
        {
            "stuff": np.ndarray[1, 3]
        }
        """
        super().__init__(**kwargs)
        self.is_sequenced = sequenced

    def __len__(self) -> int:
        if self.is_sequenced: 
            return self._sequenced_len()
        else:
            return self._non_sequenced_len()

    def __getitem__(self, index) -> dict:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__len__())
            return [self[i] for i in range(start, stop, step)] # type: ignore

        if self.is_sequenced:
            return self._sequenced_getitem(index)
        else:
            return self._non_sequenced_getitem(index)
        
    def _sequenced_getitem(self, index) -> dict:
        return self._get_input()[index].get_observations()
    
    def _non_sequenced_getitem(self, index) -> dict:
        for sampler_output_base in self._get_input():
            if index < len(sampler_output_base):
                return sampler_output_base[index]
            index -= len(sampler_output_base)
        raise IndexError(f"Index {index} out of range for Flatten {self.name}")

    def _sequenced_len(self):
        return len(self._get_input())
    
    def _non_sequenced_len(self):
        return sum([SamplerOutputBase.length_of_first_numpy_array_in_dict(sampler_output_base.get_observations()) for sampler_output_base in self._get_input()])
