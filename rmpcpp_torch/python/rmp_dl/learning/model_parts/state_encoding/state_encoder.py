
from typing import Optional
from rmp_dl.learning.model_parts.ray_auto_encoding.encoder import RayEncoder
from rmp_dl.learning.model_parts.state_encoding.direction_ray_encoder import DirectionRayEncoder
from rmp_dl.learning.model_parts.state_encoding.positional_encoder import PositionalEncoder
import torch
import torch.nn as nn
import pytorch_lightning as pl

class StateEncoder(nn.Module):
    def __init__(self,
                 encoding_type: str,  # none, nerf_positional, ray_encoding
                 encoding_type_parameters: dict, # Parameters for each encoding type)
                 disable_velocity: bool,
                 shared_ray_encoder: Optional[RayEncoder]=None,
    ):
        super().__init__()

        self.disable_velocity = disable_velocity

        input_state_size = 1 if self.disable_velocity else 2  # 

        # This is unnecessarily convoluted, TODO: Rewrite this
        if encoding_type == "none":
            self.model = nn.Identity()
            self.output_size = input_state_size * 3  # State is 3 dimensional
            self.forward_call = self._concat_forward

        elif encoding_type == "unit_vector_norm_separate":
            self.model = nn.Identity()
            self.output_size = input_state_size * 4  # State is now 4 dimensional
            self.forward_call = self._unit_norm_separate_forward

        elif encoding_type == "nerf_positional":
            params = encoding_type_parameters[encoding_type]
            self.model = PositionalEncoder(params["L"])
            self.output_size = 2 * input_state_size * 3 * params["L"]
            self.forward_call = self._concat_forward

        elif encoding_type == "ray_encoding_learned":
            if shared_ray_encoder is None:
                raise ValueError("shared_ray_encoder must be provided for ray_encoding_learned")
            
            params = encoding_type_parameters[encoding_type]
            self.model = DirectionRayEncoder(**params, ray_encoder=shared_ray_encoder)
            
            self.output_size = self.model.get_output_size() * input_state_size
            self.forward_call = self._ray_encoding_forward
            
        else:
            raise ValueError(f"Unknown encoding type in state encoder {encoding_type}")

    def get_output_size(self):
        return self.output_size

    def forward(self, rel_pos, vel) -> torch.Tensor:
        # Calls one of the 2 forward methods below based on the encoding type
        return self.forward_call(rel_pos, vel)

    def _concat_forward(self, rel_pos, vel):
        if self.disable_velocity:
            state = rel_pos
        else:
            state = torch.cat([rel_pos, vel], dim=-1)
        return self.model(state)
    
    def _ray_encoding_forward(self, rel_pos, vel):
        latent_rel_pos = self.model.forward(rel_pos)
        if self.disable_velocity:
            return latent_rel_pos
        latent_vel = self.model(vel)
        return torch.cat([latent_rel_pos, latent_vel], dim=-1)
    
    def _unit_norm_separate_forward(self, rel_pos, vel):
        # Get the unit norm of the relative position and velocity
        rel_pos_norm = torch.norm(rel_pos, dim=-1, keepdim=True, )
        vel_norm = torch.norm(vel, dim=-1, keepdim=True)

        # Builtin normalizers have epsilon. A bit of double work going on as we 
        # also actually compute the norm above, but this is not that big a deal
        rel_pos = torch.nn.functional.normalize(rel_pos, dim=-1)
        vel = torch.nn.functional.normalize(vel, dim=-1)

        # Concatenate the norm to the normalized vectors
        rel_pos = torch.cat([rel_pos_norm, rel_pos], dim=-1)
        vel = torch.cat([vel_norm, vel], dim=-1)

        # Concat step (and turns off velocity if disabled)
        return self._concat_forward(rel_pos, vel)
