import os, sys
from typing import Any, Callable, Dict, List, Optional

from rmp_dl.planner.observers.observer import Observer

import numpy as np
from rmp_dl.planner.state import State
import torch
from nvbloxLayerBindings import TsdfLayer
from kernelBindings import get_rays
from rmp_dl.planner.planner_params import RayObserverParameters


class RayObserver(Observer):
    def __init__(self, 
                 tsdf: TsdfLayer, 
                 ray_observer_params: RayObserverParameters,
                 observation_callback: Optional[Callable[[str, Dict], None]] = None,
                 ): #type:ignore
        super().__init__()
        self.params = ray_observer_params
        self.tsdf = tsdf

        # Callback to send data back 
        self.observation_callback = observation_callback

    def _get_observation(self, state: State) -> torch.tensor:
        rays = RayObserver.get_rays(self.tsdf, state.pos, self.params, cpu=False)

        if self.observation_callback is not None:
            # When we are rolling out policies, we are collecting a lot of data before we can start training.
            # So we convert to numpy array in that case
            self.observation_callback("rays", {"rays": rays.cpu().numpy()})

        return rays
        
    
    @staticmethod 
    def get_rays(tsdf: TsdfLayer, position: np.ndarray, 
                 params: RayObserverParameters, cpu=False) -> torch.tensor:
        assert position.shape == (3,), f"{position.shape} != (3,)"
        
        data = torch.zeros([params.N_sqrt * params.N_sqrt], requires_grad=False).to(torch.device("cuda:0"), torch.float32)

        position = np.array(position, dtype=np.float32)
        get_rays(data, position, tsdf, **params.__dict__)

        if cpu:
            data = data.cpu()

        return data