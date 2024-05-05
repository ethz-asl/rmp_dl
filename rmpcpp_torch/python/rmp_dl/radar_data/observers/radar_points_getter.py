
from typing import Callable, Dict, Optional
import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import PyState, State
from rmp_dl.radar_data.radar_data import RadarData
import torch

class RadarPointsGetter(Observer):
    def __init__(self, 
                radar_data: RadarData, 
                observation_callback: Optional[Callable[[str, Dict], None]] = None):
        super().__init__()
        self._radar_data = radar_data
        self._observation_callback = observation_callback
    
    def _get_observation(self, state: State) -> torch.Tensor:
        if not isinstance(state, PyState):
            raise ValueError("RadarPointsGetter only supports PyState. Do not use this observer in combination with a c++ planner")
        
        if state.idx is None:
            raise ValueError("State does not have an index")

        radar_points = self._radar_data.get_radar_points_at(state.idx)

        if self._observation_callback:
            self._observation_callback("radar_step_points", {"radar_points": radar_points})

        return torch.from_numpy(radar_points).to(torch.float32).to(torch.device("cuda"))
        
