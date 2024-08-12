from __future__ import annotations
import abc
import logging
import queue
from typing import Iterable, Iterator, List, Optional, final, Any
from rmp_dl.learning.data.pipeline.multiprocess_communication import WandbLoggerCommunicationTypes
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil

import torch
import wandb

logger = logging.getLogger("rmpcpp_torch")
class IndexingNotImplementedError(NotImplementedError): ...
class NodeOrderingError(RuntimeError): ...

class PipelineObjectBase(abc.ABC):

    def __init__(self, name: str, experiment_type: str, dry_run: bool = False):
        """Base class for any pipeline object

        Args:
            name (str): Name of the object
            experiment_type (str): "train", "validation", "test"
            dry_run (bool, optional): In a dry run we don't do any work, just print out information. Defaults to False.
        """
        self.name = name
        self.dry_run = dry_run
        self.experiment_type = experiment_type
        self.inputs = []
        self.model_args = None
        self.model_state_dict = None
        self.current_epoch: int = -1  # -1 denotes that we are not in an epoch yet
        # We can only use the seed after set_seed_multiprocessing is called. So in derivates use the seed only in setup() and not in the constructor
        self.seed = None

    def add_input(self, block):
        self.inputs.append(block)

    def set_seed_multiprocessing(self):
        self.seed = MpUtil.get_seed()

    def teardown(self): 
        """Tear down at the end of training
        """
        ... # We don't propagate this upwards, as the datapipeline wrapper class calls this for every node

    def clear_data(self): 
        """Clear all memory of the node and nodes upstream. 
        Mainly used by the cache on disk node to clear upstream memory after writing to disk
        """
        for block in self._get_inputs():
            self.log_debug(f"Starting clear for {block.experiment_type} - {block.name}")
            block.clear_data()
            self.log_debug(f"Finished clear for {block.experiment_type} - {block.name}")
        
    def setup(self): 
        """Setup before training starts
        """
        for block in self._get_inputs():
            self.log_debug(f"Starting setup for {block.experiment_type} - {block.name}")
            block.setup()
            self.log_debug(f"Finished setup for {block.experiment_type} - {block.name}")
    
    def on_epoch_start(self):
        """Called at the start of each epoch
        """
        for block in self._get_inputs():
            self.log_debug(f"Starting pre iteration tasks for {block.experiment_type} - {block.name}")
            block.on_epoch_start()
            self.log_debug(f"Finished pre iteration tasks for {block.experiment_type} - {block.name}")

    # Derived classes should override __iter__ if __getitem__ is not implemented
    def __iter__(self) -> Iterator[Any]: 
        # If we end up here, it means that the base class has not implemented __iter__. We can still iterate through
        # if the derived class has implemented __getitem__ and __len__
        try:
            for i in range(len(self)):
                yield self[i]
        except NotImplementedError: 
            raise NotImplementedError("Neither __iter__ nor __getitem__ is implemented for this class")
    
    def __len__(self) -> int:
        # If we have a single input we just return the trivial case
        if len(self.inputs) == 1:
            return len(self.inputs[0])
        raise IndexingNotImplementedError("Not implemented for this class. You should probably use __iter__ instead.")

    def __getitem__(self, index) -> Any:
        # If we have a single input we just return the trivial case
        if len(self.inputs) == 1:
            return self.inputs[0][index]
        raise IndexingNotImplementedError("Not implemented for this class. You should probably use __iter__ instead.")
    
    def set_model_args_and_state_dict(self, model_args, state_dict):
        self.model_args = model_args
        self.model_state_dict = state_dict

    def set_current_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    @final
    def _get_inputs(self) -> List[PipelineObjectBase]:
        return self.inputs
    
    @final
    def _get_input(self) -> PipelineObjectBase:
        if len(self.inputs) != 1:
            raise ValueError("Only a single input should be set")
        return self.inputs[0]

    @staticmethod
    def log_debug(data):
        PipelineObjectBase.log(data, level=logging.DEBUG)
    
    @staticmethod
    def log_info(data):
        PipelineObjectBase.log(data, level=logging.INFO)

    @staticmethod
    def log(data, level=logging.DEBUG):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            logger.log(level, data)
        else:
            # The dataset object is the DtaPipelineMpWrapper in this case
            worker_info.dataset.logging_queue.put((level, f"id: {str(worker_info.id)} " + str(data)))

    @staticmethod
    def wandb_log(data, **kwargs):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            if wandb.run is not None:
                wandb.log(data, **kwargs)
        else:
            # The dataset object is the DtaPipelineMpWrapper in this case
            worker_info.dataset.wandb_logging_queue.put((WandbLoggerCommunicationTypes.NORMAL, {**{"data": data}, **kwargs}))

    @staticmethod
    def wandb_log_artifact(artifact_or_path, **kwargs):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            if wandb.run is not None:
                wandb.run.log_artifact(artifact_or_path, **kwargs)
        else:
            # The dataset object is the DtaPipelineMpWrapper in this case
            worker_info.dataset.wandb_logging_queue.put((WandbLoggerCommunicationTypes.ARTIFACT, {**{"artifact_or_path": artifact_or_path}, **kwargs}))

