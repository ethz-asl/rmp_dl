
from typing import List
from rmp_dl.learning.data.pipeline.data_pipeline import DataPipeline
from rmp_dl.learning.data.pipeline.data_pipeline_mp_wrapper import DataPipelineMpWrapper
from rmp_dl.learning.data.pipeline.graphviz.graph import DataPipelineGraphviz
from rmp_dl.learning.data.utils.validation_dataset_names import ValidationDatasetNames
import wandb

class ValidationDatasets:
    def __init__(self, config, num_workers, 
                 manager,
                 config_path,
                 temporary_storage_path,
                 dataset_long_term_storage_path,
                 dataset_short_term_caching_path,
                 open3d_renderer_container_path_or_name):
        self.datasets: List[DataPipelineMpWrapper] = []

        if not config["datasets"]:
            return

        for dataset_config in config["datasets"]:
            name = dataset_config["name"]
            blocks = dataset_config["blocks"]
            requires_rollout_workers_after_setup = dataset_config["requires_rollout_workers_after_setup"] if "requires_rollout_workers_after_setup" in dataset_config else True
            max_workers = dataset_config["max_workers"] if "max_workers" in dataset_config else None

            dataset = DataPipelineMpWrapper(name=name, blocks=blocks, manager=manager, num_workers=num_workers,
                                            requires_rollout_workers_after_setup=requires_rollout_workers_after_setup,
                                            max_workers=max_workers,
                                            config_path=config_path,
                                            temporary_storage_path=temporary_storage_path,
                                            dataset_long_term_storage_path=dataset_long_term_storage_path,
                                            dataset_short_term_caching_path=dataset_short_term_caching_path,
                                            experiment_type="validation", 
                                            open3d_renderer_container_path_or_name=open3d_renderer_container_path_or_name)
            self.datasets.append(dataset)
            ValidationDatasetNames.append(name)

            self._log_dataset_info_to_wandb(dataset, name)

    def setup(self):
        for dataset in self:
            dataset.setup()
    
    def set_model_and_epoch(self, model, epoch):
        for dataset in self:
            dataset.set_model_and_epoch(model, epoch)

    def teardown(self):
        for dataset in self:
            dataset.teardown()

    def _log_dataset_info_to_wandb(self, dataset, name):
        if wandb.run is None:
            return
        
        pdv = DataPipelineGraphviz(dataset.pipeline)
        wandb.log({f"{name}-graph": wandb.Html(pdv.get_html(), inject=False)})
        
    def __iter__(self):
        return iter(self.datasets)

    def pre_iteration_task(self, model, epoch):
        for dataset in self:
            dataset.pre_iteration_task(model, epoch)
