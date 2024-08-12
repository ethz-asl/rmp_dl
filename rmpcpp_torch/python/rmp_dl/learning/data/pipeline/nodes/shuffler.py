
import numpy as np
from rmp_dl.learning.data.pipeline.pipeline_object_base import IndexingNotImplementedError, NodeOrderingError, PipelineObjectBase


class Shuffler(PipelineObjectBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_rng: np.random.Generator = None # type: ignore -> We set this in setup, as only then will the seed be set when we are multiprocessing
        self.indices = []

    def setup(self):
        super().setup()
        self.np_rng  = np.random.default_rng(self.seed)
        self._update_indices()
    
    def on_epoch_start(self):
        super().on_epoch_start()
        self._update_indices()
    
    def _update_indices(self):
        self.indices = list(range(len(self._get_input())))
        self.np_rng.shuffle(self.indices)
        self.log_debug(f"First 10 indices of dataset after shuffling: {self.indices[:10]}")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        try:
            return self._get_input()[self._map_index(index)]
        except IndexingNotImplementedError:
            raise NodeOrderingError("Shuffler expects a node that supports indexing as input.")
    
    def _map_index(self, index):
        return self.indices[index]


    