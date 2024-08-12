
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from rmp_dl.learning.data.pipeline.nodes.inflation_cycler import InflationCycler
from rmp_dl.learning.data.pipeline.nodes.length_bucketer import LengthBucketer
from rmp_dl.learning.data.pipeline.nodes.sequence_subsample_augmenter import SequenceSubsampleAugmenter
from rmp_dl.learning.data.pipeline.nodes.shuffler import Shuffler
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.pipeline.nodes.output import Output
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.pipeline.nodes.output import Output
from rmp_dl.learning.data.pipeline.nodes.every_epoch import EveryEpoch
from rmp_dl.learning.data.pipeline.nodes.flatten import Flatten
from rmp_dl.learning.data.pipeline.nodes.log_successrates_wandb import LogSuccessRatesWandb
from rmp_dl.learning.data.pipeline.nodes.open3d_rendering import Open3dRendering
from rmp_dl.learning.data.pipeline.nodes.resampler import Resampler
from rmp_dl.learning.data.pipeline.nodes.cache_on_disk import CacheOnDisk
from rmp_dl.learning.data.pipeline.nodes.samplers.position_sampler import PositionSampler
from rmp_dl.learning.data.pipeline.nodes.samplers.rollout_sampler import ExpertRolloutSampler, LearnedRolloutSampler
from rmp_dl.learning.data.pipeline.nodes.worlds.probabilistic_world import ProbabilisticWorld
from rmp_dl.learning.data.pipeline.nodes.worlds.manual_cube_world import ManualCubeWorld
from rmp_dl.learning.data.pipeline.nodes.concat import Concat
from rmp_dl.learning.data.pipeline.nodes.sweep import Sweep
import torch


NestedListOrDict = Union[Dict[str, Any], List['NestedListOrDict']]

class DataPipeline(torch.utils.data.Dataset):
    def __init__(self, 
                # We support nested lists as well. Should be the same end result
                # but the list of lists is easier to write in yaml, as we can reference other lists
                 blocks: NestedListOrDict,
                 experiment_type: str, 
                 temporary_storage_path: str,
                 config_path: str,
                 dataset_long_term_storage_path: Optional[str] = None,
                 dataset_short_term_caching_path: Optional[str] = None,
                 open3d_renderer_container_path_or_name: Optional[str] = None,
                 dry_run: bool=False, 
                 ):
        """Data pipeline wrapping the different blocks.

        Args:
            blocks (NestedListOrDict): (possibly nested) list of dicts, where each dict is a block in the pipeline
            experiment_type (str): Experiment type: "train", "validation", "test"
            temporary_storage_path (Optional[str], optional): Path to temporary storage. Because we have a multiprocessed datapipeline, 
                with some things like storing a model, or storing open3d images happening asynchronously in different processes before being used, 
                we use this temporary storage to store the data. Gets cleared after training. 
            dataset_long_term_storage_path (Optional[str], optional): Path to the long term storage of the dataset. See CacheOnDisk node for more info. 
                Defaults to None.
            dataset_short_term_caching_path (Optional[str], optional): Path to the short term caching of the dataset. See CacheOnDisk node for more info. 
                Defaults to None.
            open3d_renderer_container_path_or_name (Optional[str], optional): Path of the open3d rendering container for logging. Defaults to None.
            dry_run (bool, optional): If true, just prints out which worlds it would create and nothing else. Defaults to False.
        """
        self.blocks: Dict[str, PipelineObjectBase] = {}
        self.output: Output = None # type: ignore ->we raise an exception later on if not set
        self.experiment_type = experiment_type
        self.dry_run = dry_run
        self.temporary_storage_path = temporary_storage_path
        self.dataset_long_term_storage_path = dataset_long_term_storage_path
        self.dataset_short_term_caching_path = dataset_short_term_caching_path
        self.config_path = config_path

        self.open3d_renderer_container_path_or_name = open3d_renderer_container_path_or_name


        def flatten(lst: NestedListOrDict) -> List[Dict[str, Any]]:
            if not isinstance(lst, list):
                return [lst]
            result = []
            for el in lst:
                result.extend(flatten(el))
            return result
        
        flattened_blocks: List[Dict[str, Any]] = flatten(blocks)

        # 1st pass: create all blocks
        for block_config in flattened_blocks:
            name = block_config["name"]
            
            if name in self.blocks:
                raise ValueError(f"Block name {name} already exists.")
            
            if "parameters" not in block_config or block_config["parameters"] is None:
                block_config["parameters"] = {}
            
            params = block_config["parameters"]

            self._add_internal_params(params, name)
            
            self.blocks[name] = self._resolve_block(block_config["type"], params)

            if block_config["type"] == "Output":
                if self.output is not None:
                    raise ValueError("Only one output block is allowed.")
                self.output = self.blocks[name] # type: ignore

        if self.output is None: 
            raise ValueError("No output block is defined.")

        # 2nd pass: connect all blocks
        for block_config in flattened_blocks:
            name = block_config["name"]
            if "inputs" in block_config:
                for input_name in block_config["inputs"]:
                    self.blocks[name].add_input(self.blocks[input_name])

    def setup(self):
        for block in self.blocks.values():
            # This takes care of making the seed across workers unique in case we are multiprocessing
            # In case we are not multiprocessing, the seed is 0
            block.set_seed_multiprocessing()
        self.output.setup()

    def _add_internal_params(self, params, name):
        internal_params = {
            "experiment_type": self.experiment_type,
            "dry_run": self.dry_run,
            "name": name,
        }

        for key, value in internal_params.items():
            if key in params:
                raise ValueError(f"{key} is reserved for internal use.")
            params[key] = value


    def _resolve_block(self, block_type: str, params: Dict[str, Any]):
        if block_type == "PositionSampler":
            return PositionSampler(**params)
        elif block_type == "Output":
            return Output(**params)
        elif block_type == "Sweep":
            return Sweep(**params)
        elif block_type == "Concat":
            return Concat(**params)
        elif block_type == "CacheOnDisk":
            if self.dataset_short_term_caching_path is None:
                raise ValueError("dataset_short_term_caching_path must be set to use CacheOnDisk")
            return CacheOnDisk(**params, dataset_long_term_storage_path=self.dataset_long_term_storage_path,
                               dataset_short_term_caching_path=self.dataset_short_term_caching_path)
        elif block_type == "Resampler":
            return Resampler(**params)
        elif block_type == "Flatten":
            return Flatten(**params)
        elif block_type == "ExpertRolloutSampler":
            return ExpertRolloutSampler(**params, config_path=self.config_path)
        elif block_type == "LearnedRolloutSampler":
            return LearnedRolloutSampler(temporary_storage_path=self.temporary_storage_path, config_path=self.config_path, **params)
        elif block_type == "EveryEpoch":
            return EveryEpoch(**params)
        elif block_type == "ManualCubeWorld":
            return ManualCubeWorld(**params)
        elif block_type == "Open3dRendering":
            if self.open3d_renderer_container_path_or_name is None:
                raise ValueError("open3d_renderer_container_path_or_name must be set to use Open3dRendering")
            return Open3dRendering(**params, temporary_storage_path=self.temporary_storage_path, container_name_or_path=self.open3d_renderer_container_path_or_name)
        elif block_type == "ProbabilisticWorld":
            return ProbabilisticWorld(**params)
        elif block_type == "LogSuccessRatesWandb":
            return LogSuccessRatesWandb(**params)
        elif block_type == "Shuffler":
            return Shuffler(**params)
        elif block_type == "SequenceSubsampleAugmenter":
            return SequenceSubsampleAugmenter(**params)
        elif block_type == "LengthBucketer":
            return LengthBucketer(**params)
        elif block_type == "InflationCycler":
            return InflationCycler(**params)
        else:
            raise ValueError(f"Unknown block type {block_type}")
    
    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        return self.output[idx]

    def __iter__(self):
        yield from self.output
    
    def teardown(self):
        for block in self.blocks.values():
            block.teardown()

    def set_model_args_and_state_dict(self, model_args, state_dict):
        for block in self.blocks.values():
            block.set_model_args_and_state_dict(model_args, state_dict)

    def set_current_epoch(self, num):
        for block in self.blocks.values():
            block.set_current_epoch(num)

    def on_epoch_start(self):
        self.output.on_epoch_start()

    @property
    def current_epoch(self):
        return self.output.current_epoch
