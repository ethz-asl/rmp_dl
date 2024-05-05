from typing import Optional, Tuple, Union
from rmp_dl.learning.model_parts.output_network import OutputNetwork
from rmp_dl.learning.model_parts.ray_auto_encoding.decoder import RayDecoder
from rmp_dl.learning.model_parts.ray_auto_encoding.encoder import RayEncoder
from rmp_dl.learning.model_parts.ray_network import RayNetwork
from rmp_dl.learning.model_parts.state_encoding.halton_decoding import Halton2dDecoder, Halton2dDecoderFactory
from rmp_dl.learning.model_parts.state_network import StateNetwork
import torch
import torch.nn as nn


class RayModel(nn.Module):
    def __init__(self,
                 raynetwork: dict,  
                 statenetwork: dict,
                 outputnetwork: dict, 
                 shared_ray_encoder: dict,
                 shared_ray_decoder: dict,  # Actually not shared (but might be in the future?)
                 norm_type: str,
                 dropout: float,
                 disable_rays: bool, 
                 maximum_ray_length: float,
                 ):
        """Main model class for the ray model

        Args:
            raynetwork (dict): Paramaters for the ray network
            statenetwork (dict): Parameters for the state network
            outputnetwork (dict): Parameters for the output network
            shared_ray_encoder (dict): Parameters for the shared ray encoder
            shared_ray_decoder (dict): Parameters for the shared ray decoder
            disable_rays (bool): Whether to disable the ray network
            maximum_ray_length (float): Maximum ray length as used in the data collection policy
            norm_type (str): Type of normalization to use. Can be "batch_norm" or "layer_norm"
        """
        super().__init__()
        self.norm_type = norm_type
        self.disable_rays = disable_rays

        # At some point this was also used to encode the state, but not used anymore
        shared_ray_encoder_model = self._get_ray_encoder(**shared_ray_encoder, dropout=dropout)

        ray_decoder_model = None
        self.ray_decoding_learned = False
        if outputnetwork["decoder"]["decoding_type"] == "ray_decoding_learned":
            self.ray_decoding_learned = True
            # Only load the decoder if we actually need it
            ray_decoder_model = self._get_ray_decoder(**shared_ray_decoder, dropout=dropout)

        if not self.disable_rays:
            self.raynetwork = RayNetwork(**raynetwork, 
                                         shared_ray_encoder=shared_ray_encoder_model,
                                         maximum_ray_length=maximum_ray_length)

        self.statenetwork = StateNetwork(**statenetwork, norm_type=norm_type, dropout=dropout,
                                         shared_ray_encoder=shared_ray_encoder_model, maximum_ray_length=maximum_ray_length)

        combined_input = self.statenetwork.get_output_size() if disable_rays else self.statenetwork.get_output_size() + self.raynetwork.get_output_size()
        self.output = OutputNetwork(**outputnetwork, input_size=combined_input, norm_type=norm_type, ray_decoder=ray_decoder_model, dropout=dropout)
    
        # These things only get used in case we decode into rays instead of a cartesian direction
        self._temp_ray_weights = None

    def get_output_size(self):
        return self.output.get_output_size()
    
    def set_maximum_ray_length(self, length):
        self.raynetwork.maximum_ray_length = length
        self.statenetwork.maximum_ray_length = length

    def _get_ray_encoder(self, encoder_initialization, encoder_initialization_params, dropout) -> RayEncoder:
        if encoder_initialization == "random":
            params = encoder_initialization_params[encoder_initialization]
            return RayEncoder(**params, norm_type=self.norm_type, dropout=dropout)
        else: 
            raise ValueError(f"Unknown encoder initialization method {encoder_initialization}")

    def _get_ray_decoder(self, decoder_initialization, decoder_initialization_params, dropout) -> RayDecoder:
        if decoder_initialization == "random":
            params = decoder_initialization_params[decoder_initialization]
            return RayDecoder(**params, norm_type=self.norm_type, dropout=dropout)
        else:
            raise ValueError(f"Unknown decoder initialization method {decoder_initialization}")

    def forward(self, rays, rel_pos, vel, *, robot_radius=None, hiddens: Optional[Tuple[torch.Tensor]]=None):
        """Forward method of the model
        All tensors should be shape: 
        (sequence_length, batch_size, features)
        
        If you want to do successive calls during inference, do successive calls with 1 sequence length. 
        Successive calls will always keep the hidden state!! 
        So if you want to do a new run, you have to call reset_hidden_state() first.
        """
        if len(rays.shape) != 3:
            raise ValueError(f"Rays should be of shape (sequence_length, batch_size, features), but got shape {rays.shape}")
        if len(rel_pos.shape) != 3:
            raise ValueError(f"Relative positions should be of shape (sequence_length, batch_size, features), but got shape {rel_pos.shape}")
        if len(vel.shape) != 3:
            raise ValueError(f"Velocities should be of shape (sequence_length, batch_size, features), but got shape {vel.shape}")

        intermediate = self.statenetwork(rel_pos, vel, robot_radius)

        if not self.disable_rays:
            rel_pos_norm = torch.norm(rel_pos, dim=-1, keepdim=True)
            ray = self.raynetwork(rays, rel_pos_norm, robot_radius)
            intermediate = torch.cat([ray, intermediate], dim=-1)
        
        y, hiddens = self.output(intermediate, hiddens)
        
        return y, hiddens
    

class RayModelDirectionConversionWrapper(nn.Module):
    def __init__(self, model: RayModel, output_decoder: Optional[Union[str, Halton2dDecoder]]=None):
        super().__init__()
        self.model = model
        if output_decoder is None:
            # Default decoder 
            self.output_decoder: Halton2dDecoder = Halton2dDecoderFactory.max_sum50_decoder(self.model.get_output_size())
        elif isinstance(output_decoder, str):
            self.output_decoder: Halton2dDecoder = self.set_output_decoder_from_factory(output_decoder)
        elif isinstance(output_decoder, Halton2dDecoder):
            self.output_decoder = output_decoder
        else:
            raise ValueError(f"Unknown output decoder type {output_decoder}")

    def forward(self, *args, **kwargs):
        y, hidden = self.model(*args, **kwargs)
        if self.model.ray_decoding_learned:
            # We have to reshape the get rid of the sequence length dimension, and then put it back again
            y = self.output_decoder(y.reshape(*y.shape[1:])).reshape(*y.shape[:-1], -1)
        
        return y, hidden

    def set_output_decoder(self, output_decoder):
        self.output_decoder = output_decoder

    def set_output_decoder_from_factory(self, method):
        self.set_output_decoder(Halton2dDecoderFactory.resolve_decoder(method, self.model.get_output_size()))


class RayModelStaticSizeWrapper(nn.Module):
    def __init__(self, model, input_radius: float):
        super().__init__()
        self.model = model
        self.input_radius = input_radius

    def forward(self, rays, rel_pos, vel, robot_radius=None, hiddens: Optional[Tuple[torch.Tensor]]=None):
        if robot_radius is None:
            robot_radius = torch.full((rel_pos.shape[0], rel_pos.shape[1], 1), self.input_radius, device=rel_pos.device)
        return self.model(rays, rel_pos, vel, robot_radius, hiddens)