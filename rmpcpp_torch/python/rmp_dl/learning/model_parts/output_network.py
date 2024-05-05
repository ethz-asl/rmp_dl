
from operator import itemgetter
from typing import List, Optional, Tuple
import pytorch_lightning as pl
from rmp_dl.learning.model_parts.combined_network import CombinedNetwork
from rmp_dl.learning.model_parts.ray_auto_encoding.decoder import RayDecoder
from rmp_dl.learning.model_parts.util import ModelUtils
import torch

import torch.nn as nn

import torch.nn.functional as F

class OutputDecoder(nn.Module):
    def __init__(self, 
                 decoding_type: str,
                 decoding_type_parameters: dict,
                 ray_decoder: Optional[RayDecoder],
                 input_size: int,
                 norm_type: str,
                 dropout: float,
                 disable_grad: bool = False,
                 ):
        super().__init__()
        if decoding_type == "ray_decoding_learned":
            if ray_decoder is None:
                raise ValueError("Ray decoder cannot be None")
            
            params = decoding_type_parameters[decoding_type]
            self.connection = nn.Sequential(*ModelUtils.get_linear_leaky_sequential([input_size, ray_decoder.get_input_size()], norm_type=norm_type, dropout=dropout))
            self.decoder = ray_decoder
            self.output_size = ray_decoder.get_output_size()

        elif decoding_type == "cartesian":
            params = decoding_type_parameters[decoding_type]

            self.connection = nn.Linear(input_size, 3)
            self.output_size = 3
            self.decoder = nn.Identity()
            if params["normalize_output"]:
                self.decoder = lambda x: F.normalize(x, dim=-1)
        
        else: 
            raise ValueError(f"Unknown decoding type {decoding_type}")
        self.requires_grad_(not disable_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.connection(x)
        return self.decoder(x)

    def get_output_size(self) -> int:
        return self.output_size

class OutputNetwork(nn.Module):
    def __init__(self, 
                 combined_network: dict, # Intermediate network parameters
                 decoder: dict, # Decoder parameters
                 input_size: int, 
                 ray_decoder: Optional[RayDecoder], 
                 norm_type: str, 
                 dropout: float,
                 ):
        super().__init__()

        self.combined= CombinedNetwork(**combined_network, norm_type=norm_type, input_size=input_size, dropout=dropout)
        output_size = self.combined.get_output_size()
        
        self.decoder = OutputDecoder(**decoder, ray_decoder=ray_decoder, input_size=output_size, norm_type=norm_type, dropout=dropout)

    def forward(self, x: torch.Tensor, hiddens: Optional[Tuple[torch.Tensor]]) -> torch.Tensor:
        x, hiddens = self.combined(x, hiddens)
        return self.decoder(x), hiddens

    def get_output_size(self) -> int: 
        return self.decoder.get_output_size()
