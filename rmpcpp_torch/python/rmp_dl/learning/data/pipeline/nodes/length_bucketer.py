
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase
from rmp_dl.learning.data.pipeline.pipeline_object_base import  PipelineObjectBase


class LengthBucketer(PipelineObjectBase):
    def __init__(self, bucket_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_size = bucket_size

    def __iter__(self):
        len_func = SamplerOutputBase.length_of_first_numpy_array_in_dict

        buffer = []
        i = 0
        for data in self._get_input():
            buffer.append(data)

            if len(buffer) == self.bucket_size:
                if i == 0:
                    self.log_debug(f"Lengths of first sorted bucket: {list(map(len_func, sorted(buffer, key=len_func, reverse=True)))}")
                yield from sorted(buffer, key=len_func, reverse=True)
                buffer = []
                i += 1
        
        if i == 0:
            self.log_debug(f"Lengths of first sorted bucket: {list(map(len_func, sorted(buffer, key=len_func, reverse=True)))}")


        # yield whatever is left
        yield from sorted(buffer, key=len_func, reverse=True)





    
