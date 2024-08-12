import copy
import queue
import time
import traceback
from typing import Any, Optional
from rmp_dl.learning.data.collate import numpy_collate
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampling_function import SamplingFunction

import torch.multiprocessing as mp

class RolloutToDatasetCommunicationTypes:
    # Used for the rollout worker to send SamplerOutputBase to the dataset process
    ROLLOUT_RESULT = 1
    # Exception raised during rollout
    EXCEPTION_RAISED_ROLLOUT = 2
    # Used for the rollout worker to send a signal that it is done sampling
    # The dataset worker thus waits until it has emptied the queue and knows to continue
    FINISHED_ROLLOUT = 3
    
    # See comments at DatasetToRolloutCommunicationTypes.SAVE_PLY
    FINISHED_SAVING_TO_PLY = 5

    # In case we put as a setting that the dataset does not require rollout after setup, we put 
    # a signal here to indicate that the rollout worker does not exist
    # such that if a mistake is made and we actually do need rollouts, 
    # the dataset worker does not wait forever and throws an exception instead
    NO_ROLLOUT_WORKER = 6
    

class DatasetToRolloutCommunicationTypes:
    # Used to kill the rollout worker before the training starts
    KILL = 0
    # Used to send a sampling function to the rollout_worker
    SAMPLING_FUNCTION = 1
    # Used to send a world constructor to the rollout_worker, which the rollout worker uses to sample
    WORLD_CONSTRUCTOR = 2
    # Used to send a signal to the rollout worker that the dataset worker is done sending over world constructors
    # The rollout worker keeps on sampling until the world_constructor queue is empty, and then sends the 
    # DONE SAMPLING signal back
    FINISHED_SENDING = 3

    # Used to send a world_constructor object of which we need to save the ply file 
    # This uses the GPU so it starts a CUDA context. Did not see this coming at first, 
    # so this is kind of added to the rollout workers' responsibilities as an afterthought
    SAVE_PLY = 4


class MainToDatasetCommunicationTypes:
    KILL = 0
    # USed to send the model to the dataloader worker. 
    MODEL_AND_CURRENT_EPOCH = 1
    START_PRE_ITERATION_TASKS = 2

class DatasetToMainCommunicationTypes:
    FINISHED_PRE_ITERATION_TASKS = 0
    EXCEPTION = 1

class WandbLoggerCommunicationTypes:
    NORMAL = 0
    ARTIFACT = 1

class SamplingFunctionWrapper:
    # Used to invalidate the sampling function
    def __init__(self, sampling_function: Optional[SamplingFunction] = None):
        self.sampling_function: Optional[SamplingFunction] = sampling_function

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.sampling_function is None:
            raise RuntimeError("Sampling function is not initialized")
        return self.sampling_function(*args, **kwds)
    
    def setup(self):
        if self.sampling_function is None:
            raise RuntimeError("Sampling function is not initialized")
        self.sampling_function.setup()

    def invalidate(self):
        self.sampling_function = None
    
    def set_sampling_function(self, sampling_function: SamplingFunction):
        self.sampling_function = sampling_function

class RolloutWorker:
    def __init__(self, 
                 rollout_to_dataset_queue, 
                 dataset_to_rollout_queue,
                 logging_queue,
                 worker_id: int
                 ):
        self.rollout_to_dataset_queue = rollout_to_dataset_queue
        self.dataset_to_rollout_queue = dataset_to_rollout_queue
        self.logging_queue = logging_queue
        self.id = worker_id

    @staticmethod
    def _rollout_worker(rollout_to_dataset_queue, 
                        dataset_to_rollout_queue, 
                        logging_queue, worker_id: int):
        try: 
            sampling_function = SamplingFunctionWrapper()
            logging_queue.put(f"Rollout worker {worker_id} started")
            while True:
                # Loop instead of blocking, has helped me in debugging some issues sometimes
                if not dataset_to_rollout_queue.empty():
                    data_type: DatasetToRolloutCommunicationTypes
                    data: Any 
                    data_type, data = dataset_to_rollout_queue.get()
                    if RolloutWorker._parse_and_execute(data_type, data, rollout_to_dataset_queue, logging_queue, worker_id, sampling_function):
                        break
                else:
                    time.sleep(0.1)
        except Exception as e: 
            rollout_to_dataset_queue.put((RolloutToDatasetCommunicationTypes.EXCEPTION_RAISED_ROLLOUT, traceback.format_exc()))

    @staticmethod
    def _parse_and_execute(data_type, data, rollout_to_dataset_queue, logging_queue, 
                           worker_id: int, sampling_function: SamplingFunctionWrapper) -> bool: 
        # If we receive a kill signal, we return True to indicate that we are done
        if data_type == DatasetToRolloutCommunicationTypes.KILL:
            logging_queue.put(f"Rollout worker {worker_id} received kill signal")
            return True
        elif data_type == DatasetToRolloutCommunicationTypes.SAMPLING_FUNCTION:
            sampling_function.set_sampling_function(data)
            sampling_function.setup() 
        elif data_type == DatasetToRolloutCommunicationTypes.FINISHED_SENDING:
            # Communicate that we are done
            rollout_to_dataset_queue.put((RolloutToDatasetCommunicationTypes.FINISHED_ROLLOUT, None))

            # Safety feature to make sure we always set the sampling function again if we start sampling again
            sampling_function.invalidate()
        elif data_type == DatasetToRolloutCommunicationTypes.WORLD_CONSTRUCTOR:
            try:
                world_constructor = data
                logging_queue.put(f"id: {worker_id} {str(world_constructor)}")
                observations, stats, info = sampling_function(world_constructor)
                logging_queue.put(f"id: {worker_id} Finished. {info}")
                observations = numpy_collate(observations)
                output = SamplerOutputBase(observations, world_constructor, stats)
                rollout_to_dataset_queue.put((RolloutToDatasetCommunicationTypes.ROLLOUT_RESULT, output))
            except Exception as e:
                # If we fail during rollout, we don't want to exit completely, so we just log the exception to the main process
                # and continue the worker loop. 
                rollout_to_dataset_queue.put((RolloutToDatasetCommunicationTypes.EXCEPTION_RAISED_ROLLOUT, traceback.format_exc()))
        elif data_type == DatasetToRolloutCommunicationTypes.SAVE_PLY:
            # world: SingleWorldBase ######## world is of type SingleWorldBase, can't hint it here due to circular imports
            location, world_constructor = data
            logging_queue.put(f"id: {worker_id} Rollout worker saving world to ply at {location}")
            world_constructor().export_to_ply(location)
            rollout_to_dataset_queue.put((RolloutToDatasetCommunicationTypes.FINISHED_SAVING_TO_PLY, None))
        else:
            raise ValueError(f"Unknown data type {data_type}")
        
        return False


    def start(self):
        self.worker = mp.Process(target=RolloutWorker._rollout_worker, args=(self.rollout_to_dataset_queue, self.dataset_to_rollout_queue, 
                                                                             self.logging_queue, self.id))
        self.worker.start()
        return self.worker
