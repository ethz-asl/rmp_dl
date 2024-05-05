import cProfile
import logging
import re
from rmp_dl.learning.parser import get_parser
from rmp_dl.learning.util import WandbLatestModelCheckpoint, WandbUtil
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from rmp_dl.learning.data.datamodule import RmpDataModule, DatamoduleEpochStartCallback

from rmp_dl.learning.lightning_module import RayLightningModule

import rmp_dl.util.io as rmp_io
from rmp_dl.learning.model_io.model_util import ModelUtil

import wandb

import warnings
# warnings.filterwarnings("ignore", ".*does not have many workers.*")

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import time

def wandb_init(run_title, config, metadata_path=None):
    dirname = os.path.dirname(__file__)
    path_name = os.path.join(dirname, '../../data/wandb') if metadata_path is None else metadata_path
    os.makedirs(path_name, exist_ok=True)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="rmp_dl",
        name=run_title if run_title else "Testrun",
        dir=path_name,
        
        # track hyperparameters and run metadata
        config={
            "build_type": "Release",
            "params": config,
        },
        save_code=True,
        settings=wandb.Settings(code_dir=rmp_io.get_rmp_root_dir()),
        #mode="offline",
    )

def logger_init(logging_path):
    os.makedirs(logging_path, exist_ok=True)
    # Set up the logger
    logger = logging.getLogger('rmpcpp_torch')
    logger.setLevel(logging.DEBUG)

    # Set up the handlers, formatters, etc.
    debug_handler = logging.FileHandler(logging_path + 'debug.log', mode='w')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter('%(process)d %(levelname)s: %(asctime)s [%(filename)s:%(lineno)d] %(message)s'))

    info_handler = logging.FileHandler(logging_path + 'info.log', mode='w')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s [%(filename)s:%(lineno)d] %(message)s'))

    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.propagate = False


def load_pretrained_model(model, model_config):
    wandb_id = model_config["partial_pretraining"]["wandb_id"]
    file = WandbUtil.download_model(wandb_id=model_config["partial_pretraining"]["wandb_id"], 
                                    version=model_config["partial_pretraining"]["version"], 
                                    download_dir=f"data/temp/models/{os.getpid()}/")
    checkpoint = torch.load(file)
    sd = checkpoint["state_dict"]
    sd = ModelUtil.fix_state_dict(sd, old_config_version=WandbUtil.get_config(wandb_id)["model"]["version"])
    keys_to_delete = []
    for key in sd.keys():
        if "filters" in model_config["partial_pretraining"] and model_config["partial_pretraining"]["filters"]:
            for f in model_config["partial_pretraining"]["filters"]:
                if f in key:
                    keys_to_delete.append(key)

    for key in keys_to_delete: 
        del sd[key]

    # Replace key names with regex
    if "regex_replace" in model_config["partial_pretraining"]:
        for pattern, repl in model_config["partial_pretraining"]["regex_replace"]:
            to_delete = []
            to_add = {}
            for key in sd.keys():
                new_key = re.sub(pattern, repl, key)

                if new_key != key:
                    to_add[new_key] = sd[key]
                    to_delete.append(key)
            sd.update(to_add)
            for key in to_delete:
                del sd[key]

    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    print(f"Loaded state dict from {wandb_id}. ")
    print("It is generally normal that some keys are missing or unexpected, if we change architecture between \
            pretrained and the current one. This is okay, as those weights are then randomly initialized. ")
    print(f"Missing keys: ")
    for key in missing_keys:
        print(key)
    print(f"Unexpected keys: ")
    for key in unexpected_keys:
        print(key)



def main(num_workers, 
         config_path, 
         train_dataset_config_name, 
         validation_dataset_config_name, 
         model_config_name,
         run_title=None, 
         model=None, 
         temporary_storage_path=None,
         dataset_long_term_storage_path=None, 
         dataset_short_term_caching_path=None,
         wandb_metadata_path=None, logging_path=None, 
         open3d_renderer_container_path_or_name=None, 
         # This is not set by the parser
         wandb_connect=True):
    """See parser.py for a full description of these arguments. We default initialize to None, as this is also
    what argparse sends when they are not overridden as a command line argument, so default initializing in the argument
    list would still require checking for None below. So the code below checks for none
    and default initializes if necessary. 
    """
    if not wandb_connect:
        print("WARNING: Not connected to wandb")

    general_config, datatrain_config, datavalidation_config, model_config = rmp_io.ConfigUtil.get_configs_train(
        config_path, train_dataset_config_name, validation_dataset_config_name, model_config_name)

    full_config = {"general": general_config, "model": model_config, "dataset_train": datatrain_config, "dataset_validation": datavalidation_config}

    temporary_storage_path = temporary_storage_path if temporary_storage_path is not None else rmp_io.get_temp_data_dir()
    logging_path = logging_path if logging_path is not None else rmp_io.resolve_directory("data/logs/")
    dataset_long_term_storage_path = dataset_long_term_storage_path if dataset_long_term_storage_path is not None \
        else rmp_io.resolve_directory("data/dataset_long_term_storage/")
    dataset_short_term_caching_path = dataset_short_term_caching_path if dataset_short_term_caching_path is not None \
        else rmp_io.resolve_directory(f"data/dataset_short_term_caching/{os.getpid()}")

    if wandb_connect:
        wandb_init(run_title, full_config, wandb_metadata_path)
        logging_path += "/" + wandb.run.id + "/"
    else: 
        logging_path += "/" + "test/"
    
    logger_init(logging_path)

    if not model:
        model = RayLightningModule(**model_config["module_parameters"], batch_size=model_config["batch_size"], training=True)
        if "partial_pretraining" in model_config and model_config["partial_pretraining"]["active"]:
            load_pretrained_model(model, model_config)

    model.to(torch.device("cuda"))
    print(model)

    datamodule = RmpDataModule(datatrain_config, datavalidation_config,
                               batch_size=model_config['batch_size'], 
                               num_workers=num_workers, 
                               temporary_storage_path=temporary_storage_path,
                               config_path=config_path,
                               dataset_long_term_storage_path=dataset_long_term_storage_path,
                               dataset_short_term_caching_path=dataset_short_term_caching_path,
                               open3d_renderer_container_path_or_name=open3d_renderer_container_path_or_name, 
                               model=model)

    bar = TQDMProgressBar()
    rollout_callback = DatamoduleEpochStartCallback(datamodule, bar)

    wandb_logger = False
    if wandb_connect:
        wandb_logger = WandbLogger(project='rmp_dl', log_model=True)
        wandb_logger.watch(model, log='all')

    checkpoint_callback_every_n_epochs = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=model_config["epochs"] // 12,
        save_weights_only=True,
    )
    wandb_latest_model_checkpoint = WandbLatestModelCheckpoint()

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        max_epochs=model_config['epochs'],
        # reload_dataloaders_every_n_epochs=1,
        callbacks=[rollout_callback, bar, checkpoint_callback_every_n_epochs, wandb_latest_model_checkpoint],
        num_sanity_val_steps=0,
        gradient_clip_val=model_config['gradient_clip'],
        #profiler='advanced'
        )
    
    t = time.time()
    trainer.fit(model, datamodule)
    t = time.time() - t
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    actual_runtime = f"{int(hours):02}h {int(minutes):02}m {int(seconds):02}s"
    if wandb_connect:
        # Wandb keeps track of runtime, but when we do tests we restart runs, and wandb continues incrementing this runtime. 
        # So we keep track of runtime here as well so that it remains the same even if we do testruns afterwards. 
        wandb.config.update({"Actual Runtime": actual_runtime})
    else:
        print("Actual Runtime: ", actual_runtime)

if __name__ == "__main__":
    torch.manual_seed(0)
    
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args), wandb_connect=True)
