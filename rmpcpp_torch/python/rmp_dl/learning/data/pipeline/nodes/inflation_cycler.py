from typing import List
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase


class InflationCycler(PipelineObjectBase):
    def __init__(self, inflation_values: List[int], **kwargs):
        super().__init__(**kwargs)
        self.inflation_values = inflation_values

    def __iter__(self):
        inp: WorldConstructor
        inflation_iterator = self._inflation_value_circular_iterator()
        for inp in self._get_input():
            if not isinstance(inp, WorldConstructor):
                raise TypeError(f"Expected WorldConstructor, got {type(inp)}")
            
            inp.inflation = next(inflation_iterator)
            yield inp

    def _inflation_value_circular_iterator(self):
        while True:
            for value in self.inflation_values:
                yield value

            
 


    