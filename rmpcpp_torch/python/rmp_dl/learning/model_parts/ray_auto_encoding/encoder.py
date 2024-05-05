from typing import Optional
from rmp_dl.learning.model_parts.util import ModelUtils
import torch
import torch.nn as nn


class RayEncoder(nn.Module):
    def __init__(self, 
                 model_type: str, 
                 model_type_parameters: dict,
                 norm_type: str, 
                 dropout: float, 
                 disable_grad: bool = False,):
        super().__init__()

        if model_type == "fully_connected":
            params = model_type_parameters[model_type]
            self.model = nn.Sequential(*list(ModelUtils.get_linear_leaky_sequential(params["layer_sizes"], norm_type=norm_type, dropout=dropout)))
            self.input_size = params["layer_sizes"][0]
            self.output_size = params["layer_sizes"][-1]
        else:   
            raise ValueError(f"Unknown model type in ray encoder {model_type}")
        
        self.requires_grad_(not disable_grad)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        return self.model(rays)

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size
