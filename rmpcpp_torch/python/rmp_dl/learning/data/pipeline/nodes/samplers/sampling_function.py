
import abc
from typing import List, Tuple
import pandas as pd

from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor

class SamplingStatistics:
    pass

class RolloutSamplingStatistics(SamplingStatistics):
    def __init__(self, df: pd.DataFrame):
        self._df = df

class SamplingFunction(abc.ABC):
    def setup(self) -> None: 
        """
        Called once before the first call to sample. Use to setup the planner and other things.
        """
        ...

    @abc.abstractmethod
    def __call__(self, world_constructor: WorldConstructor) -> Tuple[List[dict], SamplingStatistics, str]: ... 
