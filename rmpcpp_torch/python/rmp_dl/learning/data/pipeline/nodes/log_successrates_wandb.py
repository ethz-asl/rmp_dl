import copy
import os
from typing import Any, List, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampling_function import RolloutSamplingStatistics, SamplingStatistics
import rmp_dl.util.io as rmp_io
import torch
import wandb

class LogSuccessRatesWandb(PipelineObjectBase):
    def __init__(self, statistic_name, **kwargs):
        super().__init__(**kwargs)
        self.statistic_name = statistic_name

    def setup(self):
        super().setup()
        self.log()

    def log(self):
        """Logs the success rate of the rollout samplers to wandb
        """
        sampler_output: SamplerOutputBase
        success_count = 0
        total_count = 0
        # Count the number of successes and total runs
        for sampler_output in self._get_input():
            statistics: Optional[SamplingStatistics] = sampler_output.get_stats()
            if statistics and isinstance(statistics, RolloutSamplingStatistics):
                total_count += 1
                success = statistics._df["success"]
                if success.bool():
                    success_count += 1

        total_count = sum(MpUtil.gather_on_process([total_count]))
        success_count = sum(MpUtil.gather_on_process([success_count]))

        if MpUtil.is_main_process() and total_count > 0:
            self.wandb_log({f"success_rate-{self.statistic_name}": success_count / total_count})



            