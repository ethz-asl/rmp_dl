from typing import List, Optional
from rmp_dl.learning.model_parts.ray_auto_encoding.encoder import RayEncoder
from rmp_dl.learning.model_parts.state_encoding.positional_encoder import PositionalEncoder
from rmp_dl.learning.model_parts.state_encoding.state_encoder import StateEncoder
from rmp_dl.learning.model_parts.util import ModelUtils

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class StateNetwork(nn.Module):
    def __init__(self, 
                 model_type: str,
                 model_type_parameters: dict, # Parameters for each model type
                 state_encoding: dict, # Parameters for state encoding
                 rel_pos_normalization_method: str, 
                 vel_normalization_method: str,
                 dropout: float,
                 norm_type: str,
                 shared_ray_encoder: Optional[RayEncoder]=None,
                 maximum_ray_length: Optional[float]=None,
                 disable_grad: bool=False,
                 ):
        super().__init__()

        self.rel_pos_normalization_method = rel_pos_normalization_method
        self.vel_normalization_method = vel_normalization_method
        
        self.state_encoder = StateEncoder(**state_encoding, shared_ray_encoder=shared_ray_encoder)

        self.maximum_ray_length = maximum_ray_length

        input_size = self.state_encoder.get_output_size()

        if model_type == "fully_connected":
            fully_connected_layer_sizes = model_type_parameters[model_type]["layer_sizes"]
            fully_connected_layer_sizes.insert(0, input_size)
            self.model = nn.Sequential(*list(ModelUtils.get_linear_leaky_sequential(fully_connected_layer_sizes, norm_type=norm_type, dropout=dropout)))
            self.output_size = fully_connected_layer_sizes[-1]
        elif model_type == "identity":
            self.model = nn.Identity()
            self.output_size = input_size
        else:
            raise ValueError(f"Unknown model type in statenetwork {model_type}")
        self.requires_grad_(not disable_grad)
            
    def forward(self, rel_pos, vel, robot_radius) -> torch.Tensor:
        rel_pos = self.normalize(rel_pos, self.rel_pos_normalization_method, robot_radius)
        vel = self.normalize(vel, self.vel_normalization_method, robot_radius)

        latent_state = self.state_encoder(rel_pos, vel)
        return self.model(latent_state)

    def get_output_size(self):
        return self.output_size

    def normalize(self, x, method, robot_radius=None):
        if method == "none":
            return x
        elif method == "sigmoid_like":
            return StateNetwork.normalize_sigmoid_like(x)
        elif method == "unit_norm":
            return F.normalize(x, dim=-1, eps=1e-6)
        elif method == "max_ray":
            if self.maximum_ray_length is None:
                raise ValueError("Maximum ray length must be set if normalization method is max_ray")
            return x / self.maximum_ray_length
        elif method == "lin_sigm":
            return StateNetwork.normalize_linear_sigmoid(x, self.maximum_ray_length)
        elif method == "lin_sigm_robot_radius":
            if robot_radius is None:
                robot_radius = torch.zeros_like(x[..., [0]])
            # TODO: THE *16 IS HARDCODED ACCORDING TO NUMBER OF RAYS = 1024, MAKE THIS CONFIGUREABLE
            robot_radius = robot_radius + 0.02 # 2cm safety margin TODO: PUT THIS IN A PROPER PLACE
            return StateNetwork.normalize_linear_sigmoid(x, robot_radius * 16)
        else:
            raise ValueError(f"Unknown normalization method {method}")

    @staticmethod
    def normalize_sigmoid_like(x):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        scaling = 2 / (1 + torch.exp(-x_norm)) - 1
        return F.normalize(x, dim=-1) * scaling

    @staticmethod
    def normalize_linear_sigmoid(x, M):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        def linpart(x):
            return x / (2 * M)

        def sigpart(x):
            return 1 / (1 + 1 * torch.exp((-x + M) / M * 2))
        scaling = torch.where(x_norm < M, linpart(x_norm), sigpart(x_norm))
        return F.normalize(x, dim=-1) * scaling
    
if __name__ == "__main__":
    x = torch.randn(10, 3)
    norm_x = StateNetwork.normalize_sigmoid_like(x)

    dir_x = F.normalize(x, dim=-1)
    dir_norm_x = F.normalize(norm_x, dim=-1)

    # Assert direction is maintained during normalization
    assert torch.allclose(dir_x, dir_norm_x), f"Normalization is not working correctly \n {dir_x} \n {dir_norm_x}"

