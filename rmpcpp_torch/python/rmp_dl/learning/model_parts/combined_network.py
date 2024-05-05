from typing import List, Optional, Tuple
from rmp_dl.learning.model_parts.combined_network_recurrent import CombinedNetworkRecurrent
from rmp_dl.learning.model_parts.util import ModelUtils
import torch
import torch.nn as nn


class CombinedNetwork(nn.Module):
    def __init__(self, 
                 model_type: str,
                 model_type_parameters: dict,
                 norm_type: str, 
                 dropout: float,
                 input_size: int,):
        super().__init__()
        self.model_type = model_type
        self.model, self.output_size = self._resolve_model(model_type, model_type_parameters, norm_type, input_size, dropout)

    def forward(self, x, hiddens: Optional[Tuple[torch.Tensor]]):
        if self.model_type == "recurrent":
            return self.model(x, hiddens)
        if hiddens:
            raise ValueError(f"Model type {self.model_type} does not support hidden states")
        return self.model(x), None

    def get_output_size(self):
        return self.output_size

    def _resolve_model(self, model_type: str, model_type_parameters: dict, norm_type: str, input_size: int, dropout: float) -> Tuple[nn.Module, int]:
        if model_type == "fully_connected":
            return self._resolve_fully_connected_network(**model_type_parameters[model_type], norm_type=norm_type, input_size=input_size, dropout=dropout)
        elif model_type == "recurrent":
            model = CombinedNetworkRecurrent(**model_type_parameters[model_type], norm_type=norm_type, input_size=input_size, dropout=dropout)
            return model, model.get_output_size()
        else: 
            raise ValueError(f"Unknown model type {model_type}")
        
    def _resolve_fully_connected_network(self, layer_sizes, norm_type, input_size, dropout):
        layer_sizes.insert(0, input_size)
        return nn.Sequential(*list(ModelUtils.get_linear_leaky_sequential(layer_sizes, norm_type=norm_type, dropout=dropout))), layer_sizes[-1]
