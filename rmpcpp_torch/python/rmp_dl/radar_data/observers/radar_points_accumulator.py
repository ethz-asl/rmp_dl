from typing import Callable, Dict, Optional
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import PyState, State
from rmp_dl.radar_data.radar_data import RadarData
import torch

class RadarPointsAccumulator(Observer):
    def __init__(self, 
                 steps: float,
                observation_getter: Observer,
    ):
        super().__init__()
        self.steps = steps
        self._observation_getter = observation_getter
        self._points = []
    
    def _get_observation(self, state: State) -> torch.Tensor:
        points = self._observation_getter(state)

        self._points.append(points)

        if len(self._points) > self.steps:
            self._points.pop(0)
        
        stacked = torch.vstack(self._points)
        return stacked
        
