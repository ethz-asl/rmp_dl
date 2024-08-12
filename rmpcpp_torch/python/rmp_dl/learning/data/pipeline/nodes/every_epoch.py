from operator import itemgetter
from typing import Any, Dict, Generator
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase

class EveryEpoch(PipelineObjectBase):
    def __init__(self, 
                 every_k_epochs: int, 
                 setup_before_training: bool,
                 aggregate: bool = True, # We can also turn off aggregation, in which case it is just used to do a rollout every epoch, and throw away the old data
                 **kwargs
                 ):
        """This node controls when we setup upstream nodes, pretty much specifically used to control when we sample. 
        We can sample every k epochs, only sample before training, aggregate data, and all combinations of these. 
        Note that this node clears the memory of upstream nodes and copies their data to its own memory.
        For aggregating this is necessary anyway, and it is just easier to always do this. 

        Args:
            every_k_epochs (int): Whether to call setup on upstream nodes every k epochs. If k = -1, we only call setup before training starts.
            setup_before_training (bool): Whether to call setup on upstream nodes before training starts. 
            aggregate (bool, optional): Whether to aggregate data from every call to setup from the upstream nodes. 
                Note that if there is a CacheOnDisk node downstream, aggregation will always happen. TODO: Detect this and give a warning
                    Defaults to True.
        """
        super().__init__(**kwargs)
        self.every_k_epochs = every_k_epochs
        self.setup_before_training = setup_before_training
        self.aggregate = aggregate

        if self.every_k_epochs == -1 and not self.setup_before_training:
            raise ValueError("EveryEpoch node needs to either setup before training, or setup every k epochs. Both cannot be False")

        self.outputs = []

    def setup(self):
        # Do not call the super() method here, as that will propagate upstream automatically, which we don't want
        if self.setup_before_training:
            self._setup_upstream()

    def on_epoch_start(self) -> None:
        # Do not call the super() method here, as that will propagate upstream automatically, which we don't want 
        if self.every_k_epochs == -1 or self.current_epoch % self.every_k_epochs != 0:
            return
        
        self._setup_upstream()

    def __len__(self):
        return len(self.outputs)
    
    def __getitem__(self, index):
        return self.outputs[index]

    def clear_data(self):
        super().clear_data() # Propagates the call upwards
        self.outputs.clear()

    def _setup_upstream(self):
        if len(self.inputs) == 0:
            raise ValueError(f"Input not set for {self.__class__.__name__}: {self.name}")
        # The baseclass method propagates this call upwards, and does some logging
        super().setup()

        # Clear the memory if we are not aggregating
        if not self.aggregate:
            self.outputs.clear()

        # Copy the data over and clear upstream memory
        self.outputs += [x for x in self._get_input()]
        self._get_input().clear_data()
