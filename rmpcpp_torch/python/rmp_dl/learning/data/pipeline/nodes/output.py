
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase


class Output(PipelineObjectBase): 
    def __iter__(self):
        yield from self._get_input()