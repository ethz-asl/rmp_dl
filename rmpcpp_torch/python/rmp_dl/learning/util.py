import copy
import json
from pytorch_lightning import Callback
from rmp_dl.planner.planner_params import WorldgenSettings
import rmp_dl.util.io as rmp_io
import torch
import wandb

import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint

class WandbLatestModelCheckpoint(Callback):
    """Callback to save a checkpoint every epoch, which we overwrite in wandb
    Have to do it this way as the ModelCheckpoint callbacks don't really work nicely with overwriting the same file in wandb
    Also the ModelCheckpoint callbacks don't seem to work well with calling `on_save_checkpoint` if we are connected to wandb. 
    """
    def on_train_epoch_end(self, trainer, pl_module):
        if wandb.run is None: 
            return

        run_id = wandb.run.id

        path = rmp_io.resolve_directory(f"data/temp/last_epoch_checkpoints/{run_id}/model-LAST-{run_id}.ckpt")
        trainer.save_checkpoint(path)

        epoch = trainer.current_epoch
        
        artifact_name = f"model-LAST-{run_id}"

        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(path, name="model.ckpt")
        wandb.run.log_artifact(artifact, aliases=[f"last", f"e{epoch}"])
        artifact.wait()  # We block here, as deleting artifacts below can cause issues otherwise
 
        # We delete the artifact of the epoch before
        if epoch > 0:
            wandb.run.use_artifact(f"{artifact_name}:e{epoch - 1}").delete(delete_aliases=True)

class WandbModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        
        if wandb.run is None: 
            return
            
        # Log the model checkpoint to wandb
        run_id = wandb.run.id
        epoch = trainer.current_epoch

        artifact_name = f"model-{run_id}"
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(self.last_model_path, name="model.ckpt")
        wandb.run.log_artifact(artifact, aliases=[f"epoch{epoch}"])

class WandbUtil:
    @staticmethod
    def download_model(wandb_id, version, download_dir, entity="rmp_dl", project="rmp_dl"):
        prepend = "LAST-" if version == "last" else ""
        api = wandb.Api()
        file = api.artifact(f"{entity}/{project}/model-{prepend}{wandb_id}:{version}").download(root=rmp_io.resolve_directory(download_dir))
        return file + "/model.ckpt"

    # This should probably not be here
    @staticmethod
    def get_worldgen_settings_from_config(config):
        world_limits = config["worldgen"]["world_limits"]
        worldgen_settings = WorldgenSettings(world_limits=(np.array(world_limits["min"]), np.array(world_limits["max"])), 
                                         voxel_size=config["worldgen"]["voxel_size"], 
                                         voxel_truncation_distance_vox=config["truncation_distance_vox"])
        return worldgen_settings

    @staticmethod
    def get_config(wandb_id, entity="rmp_dl", project="rmp_dl"):
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{wandb_id}")
        return json.loads(run.json_config)["params"]["value"]
