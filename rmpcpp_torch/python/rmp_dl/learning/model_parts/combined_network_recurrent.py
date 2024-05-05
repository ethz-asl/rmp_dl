from typing import List, Optional, Tuple
from rmp_dl.learning.model_parts.model_with_skip import RecurrentModelWithSkipAdd
from rmp_dl.learning.model_parts.util import ModelUtils
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class CombinedNetworkRecurrent(nn.Module):
    def __init__(self, 
                 model_type: str, 
                 model_type_parameters: dict, 
                 norm_type: str,
                 input_size: int,
                 dropout: float,
                 disable_grad_non_recurrent: bool = False,
                 ):
        super().__init__()
        # List of recurrent models on which we can call the hidden state methods defined below. 
        # As it is a list, the modules are not registered as submodules, which is usually what we want
        # as they are already contained in the sequential model in the line below. 

        self.recurrent_models: List[nn.Module] = []
        self.model, self.output_size = self._resolve_model(model_type, model_type_parameters, norm_type, input_size, dropout)
        if disable_grad_non_recurrent:
            # Disable grad of everything
            self.requires_grad_(False)
            # Turn the recurrent models back on
            for model in self.recurrent_models:
                model.requires_grad_(True)

    def forward(self, x, hiddens: Optional[Tuple[torch.Tensor]]):
        if hiddens is not None and len(hiddens) != len(self.recurrent_models):
            raise ValueError(f"Expected {len(self.recurrent_models)} hidden states, got {len(hiddens)}")
        
        output_hiddens = []
        rec_counter = 0
        for model in self.model:
            if model in self.recurrent_models:
                x, hidden = model(x, hiddens[rec_counter] if hiddens is not None else None)
                output_hiddens.append(hidden)
                rec_counter += 1
            else:
                x = model(x)

        return x, tuple(output_hiddens)

    def get_output_size(self):
        return self.output_size

    def _resolve_model(self, model_type: str, model_type_parameters: dict, norm_type: str, input_size: int, dropout: float) -> Tuple[nn.Module, int]:
        if model_type == "fc_additive_lstm_matrix":
            return self._resolve_fc_additive_lstm_matrix(**model_type_parameters[model_type], norm_type=norm_type, input_size=input_size, dropout=dropout)
        elif model_type == "fc_additive_gru_matrix":
            return self._resolve_fc_additive_gru_matrix(**model_type_parameters[model_type], norm_type=norm_type, input_size=input_size, dropout=dropout)
        else: 
            raise ValueError(f"Unknown model type {model_type}")
    
    def _resolve_fc_additive_lstm_matrix(self, lstm_hidden_sizes, lstm_depths, **kwargs):
        return self._resolve_fc_additive_mem_matrix(hidden_sizes=lstm_hidden_sizes, depths=lstm_depths, memory_type="LSTM", **kwargs)
    
    def _resolve_fc_additive_gru_matrix(self, lstm_hidden_sizes, lstm_depths, **kwargs):
        return self._resolve_fc_additive_mem_matrix(hidden_sizes=lstm_hidden_sizes, depths=lstm_depths, memory_type="GRU", **kwargs)
    
    def _resolve_fc_additive_mem_matrix(self, layer_sizes: List[int], 
                                         hidden_sizes: List[int], 
                                         depths: List[int],
                                         input_size: int,
                                         norm_type: str,
                                         dropout: float,
                                         memory_type: str, 
                                         ):
        """Insert a 'matrix' of LSTMs between fully connected layers

        Args:
            layer_sizes (List[int]): Sizes of the fully connected layers
            hidden_sizes (List[int]): Sizes of the hidden states of the memory blocks
            depths (List[int]): Number of `stacked` memory blocks at each position
            input_size (int): Input size before these layers. Will make a fully connected layer with this size as input for the first layer
            batch_norm (bool): Batch norm
            memory_type (str): switch between LSTM and GRU
        """
        if len(depths) != len(layer_sizes) - 1 or len(depths) != len(hidden_sizes):
            raise ValueError("depths and hidden_sizes must have the same length as layer_sizes - 1")
        layers = []

        
        # Connection part
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        if norm_type == "batch_norm":
            layers.append(ModelUtils.PermutedBatchNorm(layer_sizes[0]))
        elif norm_type == "layer_norm":
            layers.append(nn.LayerNorm(layer_sizes[0]))
        elif norm_type == "layer_norm_no_elementwise_affine":
            layers.append(nn.LayerNorm(layer_sizes[0], elementwise_affine=False))
        elif norm_type != "none":
            raise ValueError(f"Unknown norm type {norm_type}")

        if dropout > 0.0:
            # See comments in ModelUtil class on why we wrap it in a sequential
            layers.append(nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout)))
        else:
            layers.append(nn.LeakyReLU())

        # Fully connected part with inserted mem units (or identity if no mem unit)
        for i in range(1, len(layer_sizes)):
            if depths[i - 1] > 0:
                if layer_sizes[i] == hidden_sizes[i - 1]:
                    if memory_type == "LSTM":
                        mem_unit = nn.LSTM(layer_sizes[i - 1], layer_sizes[i], depths[i - 1], dropout=dropout)
                    elif memory_type == "GRU":
                        mem_unit = nn.GRU(layer_sizes[i - 1], layer_sizes[i], depths[i - 1], dropout=dropout)
                    else: 
                        raise ValueError(f"Unknown memory type {memory_type}")
                else:
                    if memory_type == "LSTM":
                        mem_unit = nn.LSTM(layer_sizes[i - 1], hidden_sizes[i - 1], depths[i - 1], proj_size=layer_sizes[i], dropout=dropout)
                    elif memory_type == "GRU":
                        raise ValueError("Can't have GRU with different hidden and output size")
                    else:
                        raise ValueError(f"Unknown memory type {memory_type}")
                # Used to keep track of which models can have their hidden state reset
                with_skip = RecurrentModelWithSkipAdd(mem_unit)
                self.recurrent_models.append(with_skip)
                layers.append(with_skip)
            else:
                # We add an identity layer, such that the indices of the hidden layers in the state dict remain the same. 
                # This is useful for loading pretrained models
                # Because, say we have a pretrained model with a state dict like this:
                # {
                #  "some_layer0": ...
                #  "some_layer1": ...
                #  "some_layer2": ...
                # If we then add an LSTM at position 1 and 2, the state dict will look like this:
                # {
                #  "some_layer0": ...
                #  "some_layer_belonging_to_lstm1"
                #  "some_layer2": ...
                #  "some_layer_belonging_to_lstm3": ...
                #  "some_layer4": ...
                # }
                # Therefore, we need to move the indices of the linear layers around a bit before we can load the state dict of the pretrained model. 
                # It is nice that we only have to figure this out 
                # once for the most general case, instead of having to re-do it if we have one LSTM layer less inserted (or more).
                # Therefore we insert an identity layer, e.g. with identity at position 1, we may not have a state, but the indices stay the same:
                # {
                # "some_layer0": ...
                # "some_layer2": ...
                # "some_layer_belonging_to_lstm3": ...
                # "some_layer4": ...
                # }
                # So we only need to do some_layer1 -> some_layer2, some_layer2 -> some_layer4
                # And this stays the same for all cases. 

                layers.append(nn.Identity())


            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if norm_type == "batch_norm":
                layers.append(ModelUtils.PermutedBatchNorm(layer_sizes[1]))
            elif norm_type == "layer_norm":
                layers.append(nn.LayerNorm(layer_sizes[1]))
            elif norm_type == "layer_norm_no_elementwise_affine":
                layers.append(nn.LayerNorm(layer_sizes[1], elementwise_affine=False))
            elif norm_type != "none":
                raise ValueError(f"Unknown norm type {norm_type}")
            
            if dropout > 0.0:
                # See comments in ModelUtil class on why we wrap it in a sequential
                layers.append(nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout)))
            else:
                layers.append(nn.LeakyReLU())

        return nn.ModuleList(layers), layer_sizes[-1]

