import shutil
import sys, os

import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Tuple, Union

from pathlib import Path

def create_directories_from_path(path):
    head, tail = os.path.split(path)
    
    if os.path.splitext(tail)[1]:  # if tail has a file extension
        dir_path = head
    else:
        dir_path = path

    os.makedirs(dir_path, exist_ok=True)

def resolve_directory(path: Union[Path, str]) -> str:
    dir_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    dir_path /= "../../"
    dir_path /= Path(path)
    create_directories_from_path(dir_path)
    return str(dir_path.resolve())

def resolve_config_directory(path: Union[Path, str]="") -> str:
    dir_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    dir_path /= "../../../configs"
    dir_path /= Path(path)
    create_directories_from_path(dir_path)
    return str(dir_path.resolve())

class ConfigUtil:
    @staticmethod
    def replace_variables(data: dict, full_config: dict) -> dict: 
        # There's some type: ignore's here, we assume that the outermost data is a dict (which it is in the config)
        if isinstance(data, dict):
            return {k: ConfigUtil.replace_variables(v, full_config) for k, v in data.items()}
        elif isinstance(data, list):
            return [ConfigUtil.replace_variables(v, full_config) for v in data]  # type: ignore
        elif isinstance(data, str) and data.startswith('$'):
            parts = data[1:].split('/')
            value = full_config
            for part in parts: # type: ignore
                value = value.get(part)
                if value is None:
                    raise ValueError(f'Could not resolve reference: {data}')
            return ConfigUtil.replace_variables(value, full_config)  # It may be that we still need to resolve variables in the referenced value
        else:
            return data # tp
        
    @staticmethod
    def get_general_and_world_configs(overall_path=None):
        general_config = ConfigUtil.get_yaml_general_params(overall_path)
        worlds_config = ConfigUtil.get_yaml_worlds_params(overall_path)

        full_config = {
            "general": general_config,
            "worlds": worlds_config
        }
        resolved_general = ConfigUtil.replace_variables(general_config, full_config)
        resolved_worlds = ConfigUtil.replace_variables(worlds_config, full_config)
        return resolved_general, resolved_worlds

    @staticmethod
    def get_configs_train(overall_path, train_config_name, validation_config_name, model_config_name) -> Tuple[dict, dict, dict, dict]:
        datapipeline_config_train = ConfigUtil.get_yaml_data_pipeline_params_train(train_config_name, overall_path)
        datapipeline_config_validation = ConfigUtil.get_yaml_data_pipeline_params_validation(validation_config_name, overall_path)
        model_config = ConfigUtil.get_model_params(model_config_name, overall_path)

        general, worlds = ConfigUtil.get_general_and_world_configs(overall_path)
        
        full_config = {
            "general": general,
            "worlds": worlds,
            "datatrain": datapipeline_config_train,
            "dataval": datapipeline_config_validation,
            "model": model_config,
        }

        resolved_datatrain = ConfigUtil.replace_variables(datapipeline_config_train, full_config)
        resolved_dataval = ConfigUtil.replace_variables(datapipeline_config_validation, full_config)
        resolved_model = ConfigUtil.replace_variables(model_config, full_config)

        return general, resolved_datatrain, resolved_dataval, resolved_model

    @staticmethod
    def get_yaml_general_params(overall_path=None)-> dict:
        if overall_path is None:
            overall_path = resolve_config_directory("default")
        overall_path += "/general_config.yml"

        with open((overall_path), 'r') as file:
            params = yaml.safe_load(file)
            # For the general config we resolve internal references here already, as this is used by some policy default parameters throughout the code
            # for e.g. plotting that don't use the other configs. Note that this means that the general config cannot reference the other configs.
            # This is not a problem as the general config is not supposed to be used for that anyway (it should be the other way around. )
            params = ConfigUtil.replace_variables(params, {"general": params})
            return params

    @staticmethod
    def get_yaml_worlds_params(overall_path=None)-> dict:
        if overall_path is None:
            overall_path = resolve_config_directory("default")
        overall_path += "/probabilistic_worlds.yml"

        with open((overall_path), 'r') as file:
            params = yaml.safe_load(file)
            return params

    @staticmethod
    def get_model_params(model_config_name, overall_path)-> dict:
        overall_path += f"/model_config/{model_config_name}.yml"

        with open((overall_path), 'r') as file:
            params = yaml.safe_load(file)
            return params

    @staticmethod
    def get_yaml_data_pipeline_params_validation(validation_config_name, overall_path)-> dict:
        overall_path += f"/data_pipeline_config_validation/{validation_config_name}.yml"

        with open((overall_path), 'r') as file:
            params = yaml.safe_load(file)
            return params

    @staticmethod
    def get_yaml_data_pipeline_params_train(train_config_name, overall_path)-> dict:
        overall_path += f"/data_pipeline_config_train/{train_config_name}.yml"
        
        with open((overall_path), 'r') as file:
            params = yaml.safe_load(file)
            return params
        
def get_wandb_dir() -> str:
    return resolve_directory("wandb")

def get_data_dir() -> str: 
    return resolve_directory("data")

def get_data_cache_dir() -> str:
    return resolve_directory("data/cache")

def get_temp_data_dir() -> str:
    return resolve_directory("data/temp")

def get_rmp_root_dir() -> str:
    return resolve_directory("../../")

def get_3dmatch_dir() -> str:
    return resolve_directory("../lfs/3dmatch")
    
class TempDirectories:
    def __init__(self, folder_inside_temp):
        self.folder_inside_temp = folder_inside_temp

    def __enter__(self):
        self.temp_dir = get_temp_data_dir() + f"/{self.folder_inside_temp}"
        os.makedirs(self.temp_dir, exist_ok=False)
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.temp_dir)
        
