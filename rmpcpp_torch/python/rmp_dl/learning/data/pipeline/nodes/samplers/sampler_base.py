

import abc
import copy
import logging
import queue
from threading import Thread
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from rmp_dl.learning.data.collate import numpy_collate
from rmp_dl.learning.data.pipeline.multiprocess_communication import DatasetToRolloutCommunicationTypes, RolloutToDatasetCommunicationTypes
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampling_function import SamplingFunction
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor


class SamplerBase(PipelineObjectBase):
    def __init__(self, **kwargs):
        """
        Sampler node. Call setup to start the sampling process. Calling setup successively will overwrite previous samples. 

        We can control when we sample (e.g. only before training, every epoch, etc.) 
        by calling this method at the appropriate time, using the proper downstream data pipeline nodes:
        the EveryEpoch node can be used for this, and controls when setup() of upstream nodes is called, 
        clearing memory inbetween calls. 

        The default is that we sample once before training starts, as we start sampling once setup() is called, 
        which is propagated by default through the pipeline before training. 
        """
        super().__init__(**kwargs)
        self.outputs: List[SamplerOutputBase] = []
        self._output_handler_exception = None

    @abc.abstractmethod
    def _get_sampling_function(self) -> SamplingFunction: ...

    def __len__(self) -> int:
        return len(self.outputs)

    def __getitem__(self, index) -> SamplerOutputBase:
        return self.outputs[index]

    def clear_data(self):
        super().clear_data() # Propagates to upstream nodes
        # This is still important to have even though resampling results in outputs being overwritten, 
        # as if we are caching on disk downstream, we want to clear memory as soon as possible.
        self.outputs.clear()

    def setup(self):
        if self.inputs is None:
            raise ValueError("Sampler input must be set before setup")
        
        self.outputs = self._sample(self._get_sampling_function())

    def _sample(self, sampling_function: SamplingFunction) -> List[SamplerOutputBase]:
        self._get_input().on_epoch_start()
        iterable = iter(self._get_input())

        self.log_info(f"Starting sampling {self.experiment_type}")

        if MpUtil.is_multiprocessed():
            # We use separate workers to sample in case we are multiprocessing. 
            # See comments DataPipelineMpWrapper for more details 
            outputs = self.rollout_worker_sampling(iterable, sampling_function)
        else:
            # In case we are not multiprocessing, we just sample within this process
            outputs = self.single_process_sampling(iterable, sampling_function)
        
        self.log_info("Finished sampling")

        return outputs

    def rollout_worker_sampling(self, world_constructor_iterable: Iterable[WorldConstructor], sampling_function: SamplingFunction):
        # We use a rollout worker to sample, see comments in DataPipelineMpWrapper for more details 
        
        # We get the queues to communicate with the worker
        worker_input_queue = MpUtil.get_dataset_to_rollout_queue()
        worker_output_queue = MpUtil.get_rollout_to_dataset_queue()

        # First we need to send the sampling function to the worker
        worker_input_queue.put((DatasetToRolloutCommunicationTypes.SAMPLING_FUNCTION, sampling_function))
        outputs = []
        # We start a thread to get the outputs from the worker
        output_thread = Thread(target=self._output_handler, args=(worker_output_queue, outputs))
        output_thread.start()

        # Add items to input queue
        for i, world_constructor in enumerate(world_constructor_iterable):
            world_constructor = self._resolve_world_constructor(world_constructor, i)
            if world_constructor is None:
                continue
            worker_input_queue.put((DatasetToRolloutCommunicationTypes.WORLD_CONSTRUCTOR, world_constructor))
        
        # Send the signal that we are done sending
        worker_input_queue.put((DatasetToRolloutCommunicationTypes.FINISHED_SENDING, None))

        # We wait for the output handler to finish
        output_thread.join()


        if self._output_handler_exception is not None:
            raise self._output_handler_exception # This will be caught in DataPipelineMpWrapper and correctly communicated to the main process. 
        
        # Wait for all processes to finish sampling before moving on. 
        # May not be strictly necessary, but I think it's good to keep the traversal of the pipeline relatively uniform 
        # across processes. 
        MpUtil.barrier()

        return outputs

        
    def _output_handler(self, output_queue, outputs):
        try:
            while True: 
                data = output_queue.get()
                data_type, data = data
                if data_type == RolloutToDatasetCommunicationTypes.EXCEPTION_RAISED_ROLLOUT:
                    # We just skip if there was a problem with the rollout worker
                    self.log_debug("Exception raised in rollout worker. Skipping and continuing")
                    self.log_debug(data)
                elif data_type == RolloutToDatasetCommunicationTypes.ROLLOUT_RESULT:
                    output: SamplerOutputBase = copy.deepcopy(data)
                    del data
                    outputs.append(output)
                elif data_type == RolloutToDatasetCommunicationTypes.FINISHED_ROLLOUT:
                    break
                elif data_type == RolloutToDatasetCommunicationTypes.NO_ROLLOUT_WORKER:
                    raise RuntimeError("No active rollout worker. Make sure to set requires_rollout_workers_after_setup to True in the DataPipelineMpWrapper")
                else:
                    raise RuntimeError("Unexpected return type in output handler")
        except Exception as e:
            # We want to communicate to the main thread that an exception was raised and exit gracefully
            self._output_handler_exception = e

    def single_process_sampling(self, world_constructor_iterable: Iterable[WorldConstructor], sampling_function: SamplingFunction):
        sampling_function.setup()
        outputs = []

        for i, world_constructor in enumerate(world_constructor_iterable):
            world_constructor = self._resolve_world_constructor(world_constructor, i)
            if world_constructor is None:
                continue
            self.log_debug(world_constructor)
            observations, stats, info = sampling_function.__call__(world_constructor)
            self.log_debug(info)
            observations = numpy_collate(observations)
            output = SamplerOutputBase(observations, world_constructor, stats)
            outputs.append(copy.deepcopy(output))
 
        return outputs
    
    def _resolve_world_constructor(self, world_constructor, i):
        if MpUtil.is_multiprocessed() and i % MpUtil.num_workers() != MpUtil.id():
            return None

        if self.dry_run:
            self.log_debug(world_constructor)
            return None
        
        return world_constructor