import os, sys
from typing import Any, Callable, Dict, List, Optional

from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import State

class RaySaver(Observer):
    def __init__(self, 
                 name: str,
                 ray_observation_getter: Observer, 
                 observation_callback: Callable[[str, Dict], None]
                 ): 
        super().__init__()
        self.name = name
        self.ray_observation_getter = ray_observation_getter
        self.observation_callback = observation_callback

    def _get_observation(self, state: State) -> None:
        rays = self.ray_observation_getter(state)
        self.observation_callback(self.name, {"rays": rays.cpu().numpy()})