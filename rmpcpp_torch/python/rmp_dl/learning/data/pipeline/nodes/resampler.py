import abc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import logging
from typing import Any, Dict, List
import numpy as np
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase


class Resampler(PipelineObjectBase):
    def __init__(self, count, **kwargs):
        super().__init__(**kwargs)

        self.count = count

        # Set later during setup, as we are not yet multiprocessing here and this depends on the worker seed. 
        self.generator = None
        self.indices = []

    def setup(self):
        # We only set it here, as during __init__ we are not yet multiprocessing
        self.generator = np.random.default_rng(self.seed)
        super().setup()
        # If we are doing multiprocessing, we need to split the count across the workers
        # We don't know the batch size, so let's just take a large one (256), so that 
        # whatever we output should pretty much always divide nicely for smaller batch sizes

        batches256 = self.count // 256 # If this doesn't divide evenly there is nothing more we can do.

        # We now want to divide full batches evenly
        batches_per_worker, remainder = divmod(batches256, MpUtil.num_workers())
        additional_batch = 1 if MpUtil.id() < remainder else 0

        self.count = (batches_per_worker + additional_batch) * 256


    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self._set_indices()

    def _set_indices(self):
        if len(self._get_input()) == 0:
            self.indices = []
        else:
            self.indices = list(map(int, self.generator.choice(len(self._get_input()), size=self.count, replace=True)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self._get_input()[self.indices[index]]