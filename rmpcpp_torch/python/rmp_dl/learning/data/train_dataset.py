

from rmp_dl.learning.data.pipeline.data_pipeline import DataPipeline
from rmp_dl.learning.data.pipeline.data_pipeline_mp_wrapper import DataPipelineMpWrapper
from rmp_dl.learning.data.pipeline.graphviz.graph import DataPipelineGraphviz
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
import torch
import wandb

class TrainDataset(DataPipelineMpWrapper):
    # Important to do inheritance over composition here because in the validation datasets, the DataPipelineMpWrapper is the outermost dataset 
    # class too. We want the same class to be the outermost class, as this makes accessing the dataset
    #  inside workers via `worker_info = torch.utils.data.get_worker_info().dataset` function identically for 
    # the training and validation datasets. If another object wrapped the DataPipelineMpWrapper via composition for the training dataset,
    # the dataset object in get_worker_info() would be the wrapping object and would not contain the pipeline dataset object

    def __init__(self, config, **kwargs):
        requires_rollout_workers_after_setup = config["requires_rollout_workers_after_setup"] if "requires_rollout_workers_after_setup" in config else True
        max_workers = config["max_workers"] if "max_workers" in config else None
        super().__init__(blocks=config["blocks"], 
                         requires_rollout_workers_after_setup=requires_rollout_workers_after_setup,
                         max_workers=max_workers,
                         experiment_type="train",
                         **kwargs)
        self._log_dataset_info_to_wandb()

    def _log_dataset_info_to_wandb(self):
        if not MpUtil.is_main_process():
            return
            
        if wandb.run is None:
            return
        
        pdv = DataPipelineGraphviz(self.pipeline)
        
        wandb.log({"train_graph": wandb.Html(pdv.get_html(), inject=False)})

