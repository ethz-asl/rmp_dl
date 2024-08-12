from rmp_dl.learning.data.pipeline.nodes.inflation_cycler import InflationCycler
from rmp_dl.learning.data.pipeline.nodes.length_bucketer import LengthBucketer
from rmp_dl.learning.data.pipeline.nodes.sequence_subsample_augmenter import SequenceSubsampleAugmenter
from rmp_dl.learning.data.pipeline.nodes.shuffler import Shuffler
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
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_base import SamplerBase
from rmp_dl.learning.data.pipeline.nodes.worlds.probabilistic_world import ProbabilisticWorld
from rmp_dl.learning.data.pipeline.nodes.worlds.manual_cube_world import ManualCubeWorld
from rmp_dl.learning.data.pipeline.nodes.worlds.single_world_base import SingleWorldBase
from rmp_dl.learning.data.pipeline.nodes.concat import Concat
from rmp_dl.learning.data.pipeline.nodes.sweep import Sweep

class NodeFormatter:
    @staticmethod
    def format(node: PipelineObjectBase) -> str:
        if isinstance(node, Sweep):
            return NodeFormatter.format_sweep_node(node)
        elif isinstance(node, ManualCubeWorld):
            return NodeFormatter.format_manual_cube_world_node(node)
        elif isinstance(node, PositionSampler):
            return NodeFormatter.format_position_sampler_node(node)
        elif isinstance(node, Output):
            return NodeFormatter.format_output_node(node)
        elif isinstance(node, Concat):
            return NodeFormatter.format_concat_node(node)
        elif isinstance(node, CacheOnDisk):
            return NodeFormatter.format_cache_on_disk_node(node)
        elif isinstance(node, Resampler):
            return NodeFormatter.format_resampler_node(node)
        elif isinstance(node, Flatten):
            return NodeFormatter.format_flatten_node(node)
        elif isinstance(node, ExpertRolloutSampler):
            return NodeFormatter.format_expert_rollout_sampler_node(node)
        elif isinstance(node, LearnedRolloutSampler):
            return NodeFormatter.format_learned_rollout_sampler_node(node)
        elif isinstance(node, EveryEpoch):
            return NodeFormatter.format_every_epoch_node(node)
        elif isinstance(node, Open3dRendering):
            return NodeFormatter.format_open3d_rendering_node(node)
        elif isinstance(node, ProbabilisticWorld):
            return NodeFormatter.format_probabilistic_world_node(node)
        elif isinstance(node, LogSuccessRatesWandb):
            return NodeFormatter.format_log_success_rates_wandb_node(node)
        elif isinstance(node, Shuffler):
            return NodeFormatter.format_shuffler_node(node)
        elif isinstance(node, LengthBucketer):
            return NodeFormatter.format_length_bucketer_node(node)
        elif isinstance(node, SequenceSubsampleAugmenter):
            return NodeFormatter.format_sequence_subsample_augmenter_node(node)
        elif isinstance(node, InflationCycler):
            return NodeFormatter.format_inflation_cycler_node(node)
        else: 
            return "Implement node formatting in node_formatter.py"

    @staticmethod
    def get_color(node: PipelineObjectBase) -> str:
        # We go from the most specific to the most general
        if issubclass(type(node), SingleWorldBase):
            return "#e74c3c"
        elif issubclass(type(node), SamplerBase):
            return "#27ae60"
        elif issubclass(type(node), PipelineObjectBase):
            return "#3498db"
        else: 
            return "white"
    

    @staticmethod
    def format_sweep_node(node: Sweep) -> str:
        output = ""
        output += "Sweep\l"
        output += "  - name: " + node.name + "\l"
        output += "  - sweep variable name: " + node.sweep_variable_name + "\l"
        output += "  - sweep type: " + node.sweep_type + "\l"
        output += "  - sweep type params: " + "\l"
        output += "      " + str(node.sweep_type_params) + "\l"
        return output


    @staticmethod
    def format_manual_cube_world_node(node: ManualCubeWorld) -> str:
        output = ""
        output += "ManualCubeWorld\l"
        output += "  - name: " + node.name + "\l"
        output += "  - cubes: " + "\l"
        for cube in node.cubes:
            output += "      " + str(cube) + "\l"
        output += "  - world_limits: " + "\l"
        output += "  - start: " + str(node.start) + "\l"
        output += "  - goal: " + str(node.goal) + "\l"

        return output
    
    @staticmethod
    def format_position_sampler_node(node: PositionSampler) -> str:
        output = ""
        output += "PositionSampler\l"
        output += "  - name: " + node.name + "\l"
        output += "  - num_samples: " + str(node.num_samples) + "\l"
        output += "  - sampler_type: " + node.sampler_type + "\l"
        output += "  - sampler_params: " + str(node.sampler_params) + "\l"

        return output
    
    @staticmethod
    def format_cache_on_disk_node(node: CacheOnDisk) -> str:
        output = ""
        output += "CacheOnDisk\l"
        output += "  - name: " + node.name + "\l"
        output += "  - cache_name: " + node.cache_name + "\l"
        output += "  - store_permanently: " + str(node.store_permanently) + "\l"
        output += "  - chunk_size: " + str(node.chunk_size) + "\l"
        output += "  - keep_in_memory: " + str(node.keep_in_memory) + "\l"
        return output

    @staticmethod
    def format_resampler_node(node: Resampler):
        output = ""
        output += "Resampler\l"
        output += "  - name: " + node.name + "\l"
        output += "  - count: " + str(node.count) + "\l"
        output += "  - seed: " + str(node.seed) + "\l" 

        return output
        
    @staticmethod
    def format_learned_rollout_sampler_node(node: LearnedRolloutSampler):
        output = ""
        output += "LearnedRolloutSampler\l"
        output += "  - name: " + node.name + "\l"
        output += "  - stride: " + str(node.stride) + "\l"
        output += "  - terminate_when_stuck: " + str(node.terminate_if_stuck) + "\l"

        return output

    @staticmethod
    def format_expert_rollout_sampler_node(node: ExpertRolloutSampler):
        output = ""
        output += "ExpertRolloutSampler\l"
        output += "  - name: " + node.name + "\l"
        output += "  - stride: " + str(node.stride) + "\l"
        output += "  - terminate_when_stuck: " + str(node.terminate_if_stuck) + "\l"

        return output

    @staticmethod
    def format_every_epoch_node(node: EveryEpoch):
        output = ""
        output += "EveryEpoch\l"
        output += "  - name: " + node.name + "\l"
        output += "  - every_k_epochs: " + str(node.every_k_epochs) + "\l"
        output += "  - setup_before_training: " + str(node.setup_before_training) + "\l"
        output += "  - aggregate: " + str(node.aggregate) + "\l"

        return output 

    @staticmethod
    def format_open3d_rendering_node(node: Open3dRendering):
        output = ""
        output += "Open3dRendering\l"
        output += "  - name: " + node.name + "\l"

        return output
        

    @staticmethod
    def format_probabilistic_world_node(node: ProbabilisticWorld):
        output = ""
        output += "ProbabilisticWorld\l"
        output += "  - name: " + node.name + "\l"
        output += "  - seed: " + str(node.seed) + "\l"
        output += "  - start_goal_margin_to_obstacles: " + str(node.start_goal_margin_to_obstacles) + "\l"
        output += "  - start_goal_location_type: " + str(node.start_goal_location_type) + "\l"
        output += "  - start_goal_location_params: " + str(node.start_goal_location_type_params) + "\l"
        output += "  - groups: " + "\l"
        for group in node.obstacle_groups:
            output += "      " + str(group["name"]) + "\l"

        return output
    

    @staticmethod
    def format_log_success_rates_wandb_node(node: LogSuccessRatesWandb):
        output = ""
        output += "LogSuccessRatesWandb\l"
        output += "  - name: " + node.name + "\l"
        output += "  - statistic_name: " + node.statistic_name + "\l"
        return output


    @staticmethod
    def format_flatten_node(node: Flatten):
        output = ""
        output += "Flatten\l"
        output += "  - name: " + node.name + "\l"
        output += "  - sequenced: " + str(node.is_sequenced) + "\l"
        return output

    @staticmethod
    def format_output_node(node: Output) -> str:
        return "Output"
    
    @staticmethod
    def format_concat_node(node: Concat) -> str:
        return "Concat"

    @staticmethod
    def format_shuffler_node(node: Shuffler) -> str:
        return "Shuffler"

    @staticmethod
    def format_sequence_subsample_augmenter_node(node: SequenceSubsampleAugmenter) -> str:
        output = ""
        output += "SequenceSubsampleAugmenter\l"
        output += "  - name: " + node.name + "\l"
        output += "  - count: " + str(node.count) + "\l"
        output += "  - length_mean: " + str(node.length_mean) + "\l"
        output += "  - length_std: " + str(node.length_std) + "\l"

        return output

    @staticmethod
    def format_length_bucketer_node(node: LengthBucketer) -> str:
        output = ""
        output += "LengthBucketer\l"
        output += "  - name: " + node.name + "\l"
        output += "  - bucket_size: " + str(node.bucket_size) + "\l"

        return output

    @staticmethod
    def format_inflation_cycler_node(node: InflationCycler) -> str:
        output = ""
        output += "InflationCycler\l"
        output += "  - name: " + node.name + "\l"
        output += "  - inflations: " + str(node.inflation_values) + "\l"

        return output