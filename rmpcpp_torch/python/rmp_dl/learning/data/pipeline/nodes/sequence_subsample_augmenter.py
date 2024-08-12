
import numpy as np
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase
from rmp_dl.learning.data.pipeline.pipeline_object_base import IndexingNotImplementedError, NodeOrderingError, PipelineObjectBase


class SequenceSubsampleAugmenter(PipelineObjectBase):
    def __init__(self, 
                 count: int, 
                 length_mean: float, 
                 length_std: float,
                 *args, **kwargs):
        """Augments data by subsampling input sequences
        Note that this results in highly correlated data when iterating over this node. 
        However, you can't put a shuffler downstream, as this node does not implement __getitem__,
        this would be very slow in general because usually we read input sequences from disk 
        somwhere, so we iterate over inputs only. To make data not so correlated, use a
        LengthBucketer downstream, buckets elements and iterates over them according to their length. 
        Make sure to set the buffer size large enough so that enough data gets combined. 

        Args:
            count (int): Number of subsamples per input sequence
            length_mean (float): Mean length of subsample
            length_std (float): Standard dev of the length of subsample
        """
        super().__init__(*args, **kwargs)
        self.count = count
        self.length_mean = length_mean
        self.length_std = length_std
        self.np_rng: np.random.Generator = None # type: ignore -> We set this in setup, as only then will the seed be set when we are multiprocessing

    def setup(self):
        super().setup()
        self.np_rng  = np.random.default_rng(self.seed)

    def __iter__(self):
        for data in self._get_input():
            self.augment_data(data)
            yield from data

    def augment_data(self, data):
        len_func = SamplerOutputBase.length_of_first_numpy_array_in_dict
        def extract(d, start, stop):
            if isinstance(d, np.ndarray):
                return d[start:stop] 
            elif isinstance(d, dict):
                return {k: extract(v, start, stop) for k, v in d.items()}
            raise ValueError("Expected a dict or np.ndarray, got {}".format(type(d).__name__))

        new_data = []
        d_length = len_func(data)
        if d_length == 1:
            raise NodeOrderingError("Trying to augment non-sequenced data. Make sure to correctly order the data pipeline nodes")

        for _ in range(self.count):
            # We sample a length from a normal distribution, and then sample at for the middle of the sequence
            length = max(10, int(self.np_rng.normal(self.length_mean, self.length_std)))
            mid_index = self.np_rng.integers(0, d_length)

            # Clip bounds
            low = max(0, mid_index - length // 2)
            high = min(d_length, mid_index + length // 2)

            # Note that slicing np arrays does not make copies and just creates a view of the original, 
            # so this is fast and no extra memory is used. 
            sliced = extract(data, low, high)
            new_data.append(sliced) 
        data.extend(new_data)
    

    