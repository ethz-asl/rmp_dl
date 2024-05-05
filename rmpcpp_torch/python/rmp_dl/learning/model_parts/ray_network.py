
from typing import Optional
import pytorch_lightning as pl
from rmp_dl.learning.model_parts.ray_auto_encoding.encoder import RayEncoder
from rmp_dl.learning.model_parts.state_network import StateNetwork
from rmp_dl.learning.model_parts.util import ModelUtils
import torch

import torch.nn as nn


class RayNetwork(nn.Module):
    def __init__(self, 
                 ray_normalization_method: str,  # none, invert, max, goal
                 maximum_ray_length: float, 
                 shared_ray_encoder: RayEncoder, 
                 disable_grad: bool,
                 ):
        super().__init__()
        self.ray_normalization_method = ray_normalization_method
        self.maximum_ray_length = maximum_ray_length


        self.encoder = shared_ray_encoder
        self.requires_grad_(not disable_grad)

    def forward(self, rays: torch.Tensor, goal_dist: Optional[torch.Tensor]=None, robot_radius=None) -> torch.Tensor:
        if self.ray_normalization_method == "none":
            pass
        elif self.ray_normalization_method == "invert":
            rays = 1 / (1 + rays)
        elif self.ray_normalization_method == "max":
            rays = rays / self.maximum_ray_length
        elif self.ray_normalization_method == "goal":
            if goal_dist is None:
                raise ValueError("Goal distance cannot be None when using goal normalization")
            rays = rays / goal_dist
        elif self.ray_normalization_method == "lin_sigm_robot_radius":
            if robot_radius is None:
                robot_radius = torch.zeros_like(rays[..., [0]])
            # TODO: THE *16 IS HARDCODED ACCORDING TO NUMBER OF RAYS = 1024, MAKE THIS CONFIGUREABLE
            robot_radius = (robot_radius + 0.02) * 16. # 2cm safety margin TODO: PUT THIS IN A PROPER PLACE
            # We unsqueeze, as the normalize linear sigmoid method normalizes over the last dimension. 
            # We want to normalize over every single ray value, so we add a dimension to make it work
            rays = StateNetwork.normalize_linear_sigmoid(rays.unsqueeze(-1), robot_radius.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f"Unknown normalization method {self.ray_normalization_method}")
        
        return self.encoder(rays)

    def get_output_size(self):
        return self.encoder.get_output_size()


    