from typing import Tuple
from rmp_dl.learning.model_parts.ray_auto_encoding.encoder import RayEncoder
import torch
import torch.nn as nn

from rmp_dl.learning.model_parts.state_encoding.halton_encoding import Halton2dEncoder

class DirectionRayEncoder(nn.Module):
    def __init__(self, 
                 ray_encoder: RayEncoder,
                 halton_encoding: dict,
                 disable_grad: bool = False
                 ):
        """Encodes directions into latent space, see forward method.

        Args:
            model (nn.Module): Model used to convert halton rays into latent space
            halton_encoding_method (str): Method to use for halton encoding, see HaltonEncoder2d
            disable_grad (bool, optional): Disable gradient computation for the model. Defaults to False.
        """
        super().__init__()
        self.disable_grad = disable_grad 
        self.ray_encoder = ray_encoder
        self.input_size = ray_encoder.get_input_size()
        self.output_size = ray_encoder.get_output_size()
        self.direction_halton_encoder = Halton2dEncoder(self.input_size, **halton_encoding)


    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        """Encodes the given direction into a latent space
        size (batch_size, output_size)

        The encoding is done by first encoding the direction into a 2d halton sequence, and then encoding that
        into a latent space using the model given in the constructor. 

        Args:
            directions (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        # We add a dimension to the direction, as the halton encoder can take multiple directions
        # At first I wanted to pass the position and velocity together to this module,
        # but I changed my mind, I think it's more clear to have a separate module for the position and velocity
        # I wrote the encoder with the initial idea in mind and decided to keep it as it is a bit more flexible in the future
        rays = self.direction_halton_encoder(direction.unsqueeze(-1)).squeeze(-1)
        rays *= direction.norm(dim=1, keepdim=True)  # Scale the rays to the length of the direction
        
        if self.disable_grad:
            with torch.no_grad():
                return self.ray_encoder(rays)
            
        return self.ray_encoder(rays)
        
    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

