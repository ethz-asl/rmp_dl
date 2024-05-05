from typing import Any, Callable, Dict, List, Optional

from rmp_dl.planner.observers.observer import Observer

from rmp_dl.planner.state import State
import torch


class RayNoiser(Observer):
    def __init__(self, 
                 noise_std: float,
                 ray_observation_getter: Observer,
                 observation_callback: Optional[Callable[[str, Dict], None]] = None,
                 ): 
        super().__init__()
        self.noise_std = noise_std
        self.ray_observation_getter = ray_observation_getter
        self.observation_callback = observation_callback

    def _get_observation(self, state: State) -> torch.tensor:
        rays = self.ray_observation_getter(state)

        # Multiplicative noise with mean 1 and std noise_std on the rays
        rays = rays * (1 + torch.randn_like(rays) * self.noise_std)

        if self.observation_callback is not None:
            # We save the observation to the callback if it is given
            self.observation_callback("rays_noisy", {"rays": rays.cpu().numpy()})

        return rays
        
    