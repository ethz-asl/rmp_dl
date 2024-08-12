from typing import Callable, List, Optional
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from rmp_dl.learning.data.pipeline.data_pipeline_mp_wrapper import DataPipelineMpWrapper
from rmp_dl.learning.data.train_dataset import TrainDataset
from rmp_dl.learning.data.validation_datasets import ValidationDatasets
from rmp_dl.learning.lightning_module import RayLightningModule
import torch
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import TQDMProgressBar

import torch.utils.data.dataloader # For default collate_fn
import wandb 

import numpy as np
import copy

from rmp_dl.learning.data.collate import filter_collate_fn

import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)

class WorkerInitFn:
    def __call__(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
        dataset: DataPipelineMpWrapper = worker_info.dataset  # the dataset copy in this worker process
        dataset._worker__setup()


class RmpDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_config, validation_config,
                 batch_size, 
                 num_workers, 
                 temporary_storage_path,
                 config_path, 
                 dataset_long_term_storage_path=None, dataset_short_term_caching_path=None,
                 open3d_renderer_container_path_or_name=None,
                 model=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.manager = mp.Manager()

        self.train_dataset = TrainDataset(train_config, name="train", manager=self.manager, num_workers=num_workers, 
                                          config_path=config_path,
                                          temporary_storage_path=temporary_storage_path,
                                          dataset_long_term_storage_path=dataset_long_term_storage_path,
                                          dataset_short_term_caching_path=dataset_short_term_caching_path,
                                          open3d_renderer_container_path_or_name=open3d_renderer_container_path_or_name)
        self.validation_datasets = ValidationDatasets(validation_config, num_workers, 
                                                      manager=self.manager,
                                                      config_path=config_path,
                                                      temporary_storage_path=temporary_storage_path,
                                                      dataset_long_term_storage_path=dataset_long_term_storage_path,
                                                      dataset_short_term_caching_path=dataset_short_term_caching_path,
                                                      open3d_renderer_container_path_or_name=open3d_renderer_container_path_or_name)

        # If we have a model already, we use this to do some initial rollouts. 
        # This is useful for when we use pretrained models. 
        # Note that at this point, we are not yet multiprocessing, so we can just set the state dict in the pipeline, 
        # and then later when multiple processes are created, the model should be passed on correctly
        if model is not None:
            model_args, state_dict = model.get_model_args_and_state_dict()
            # We put the state dict on the cpu, to make sure we don't open a cuda context for every dataset worker
            state_dict = {k: v.cpu() for k, v in state_dict.items()}  
            self.train_dataset.pipeline.set_model_args_and_state_dict(model_args, state_dict)
            for dataset in self.validation_datasets:
                dataset.pipeline.set_model_args_and_state_dict(model_args, state_dict)


    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset.setup()
            self.validation_datasets.setup()

    def teardown(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset.teardown()
            self.validation_datasets.teardown()

    def on_train_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self.train_dataset.pre_iteration_task(pl_module, trainer.current_epoch)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self.validation_datasets.pre_iteration_task(pl_module, trainer.current_epoch)
    
    def train_dataloader(self):
        persistent_workers = False
        if self.num_workers > 0:
            persistent_workers = True
        prefetch_factor = 2 * self.batch_size if self.num_workers > 0 else 2 # Kind of weird, if it is set to 2 it won't throw an error if num_workers is 0
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, collate_fn=filter_collate_fn, prefetch_factor=prefetch_factor,
                          pin_memory=True, num_workers=self.num_workers, worker_init_fn=WorkerInitFn(), persistent_workers=persistent_workers)

        rollout_workers = self.train_dataset.start_rollout_workers()
        # The datasets only get spawned once you start looping over them. We want the setup function to run immediately, so we do that here
        # DEADLOCK CAN HAPPEN HERE: If a rollout worker gets killed by slurm, the dataset worker will hang if it needs a rollout worker. 
        # TODO: Spawn a thread in the main process here which checks for dead workers and throws an exception if one is dead
        for _ in dataloader:
            break
        self.train_dataset.post_setup(rollout_workers)
        return dataloader
        
    def val_dataloader(self):  
        dataloaders = []
        for i, dataset in enumerate(self.validation_datasets):
            persistent_workers = False
            if self.num_workers > 0:
                persistent_workers = True
            rollout_workers = dataset.start_rollout_workers()
            dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=filter_collate_fn, 
                                    pin_memory=True, num_workers=dataset.num_workers, 
                                    worker_init_fn=WorkerInitFn(), persistent_workers=persistent_workers)
            # The datasets only get spawned once you start looping over them. We want the setup function to run immediately, so we do that here
            # DEADLOCK CAN HAPPEN HERE: If a rollout worker gets killed by slurm, the dataset worker will hang if it needs a rollout worker. 
            # TODO: Spawn a thread in the main process here which checks for dead workers and throws an exception if one is dead
            for _ in dataloader:
                break
            dataloaders.append(dataloader)
            dataset.post_setup(rollout_workers)

        return dataloaders

class DatamoduleEpochStartCallback(pl.Callback):
    def __init__(self, datamodule: RmpDataModule, progress_bar: Optional[TQDMProgressBar] = None):
        super(DatamoduleEpochStartCallback).__init__()
        self.datamodule = datamodule
        self.bar = progress_bar
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.datamodule.on_train_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.datamodule.on_validation_epoch_start(trainer, pl_module)


