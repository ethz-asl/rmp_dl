
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase


class Concat(PipelineObjectBase):
    def __len__(self):
        return sum([len(inp) for inp in self._get_inputs()])
    
    def __getitem__(self, index):
        i = index
        for inp in self._get_inputs():
            if i < len(inp):
                return inp[i]
            i -= len(inp)
        raise IndexError("Index out of range")
    
    def __iter__(self):
        for inp in self._get_inputs():
            yield from inp
 


    