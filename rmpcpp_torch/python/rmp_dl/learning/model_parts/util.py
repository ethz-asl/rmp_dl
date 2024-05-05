from typing import Callable, List
import torch
import torch.nn as nn

class ModelUtils:
    
    class PermutedBatchNorm(nn.BatchNorm1d):
        def forward(self, x):
            # Because batch norm expects the batch dimension to be first and the sequence length to be last, we need to permute the dimensions
            x = torch.permute(x, (1, 2, 0))
            x = super().forward(x)
            x = torch.permute(x, (2, 0, 1))
            return x

    @staticmethod
    def get_linear_leaky_sequential(layers: List[int], dropout, norm_type="none"):
        for layer in zip(layers[:-1], layers[1:]):
            yield nn.Linear(*layer)
            if norm_type == "batch_norm":
                yield ModelUtils.PermutedBatchNorm(layer[1])
            elif norm_type == "layer_norm":
                yield nn.LayerNorm(layer[1])
            elif norm_type == "layer_norm_no_elementwise_affine":
                yield nn.LayerNorm(layer[1], elementwise_affine=False)
            elif norm_type != "none":
                raise ValueError(f"Unknown norm type {norm_type}")

            if dropout > 0.0: 
                # We put relu and dropout in a sequential, as this makes it easier to potentially use pretrained networks 
                # with or without dropout, for a new network with or without dropout. 
                # As both of these layers don't have parameters, but if we put dropout in the outer sequential, 
                # it shifts the indices of the other layers, which means loading state dicts becomes a pain as you now have to 
                # shift all of the indices of the layers after the dropout layer. 
                # E.g: 
                # [0] Linear
                # [1] Leaky
                # [2] Linear
                # If we do
                # [0] Linear
                # [1] Leaky
                # [2] Dropout
                # [3] Linear  <---- Note how the index is now 3, not 2
                # We need to shift the indices of linear. 
                # If we instead do
                # [0] Linear
                # [1] Sequential(Leaky, Dropout)
                # [2] Linear  <---- Index is the same 
                yield nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout))
            else:
                yield nn.LeakyReLU()

    @staticmethod
    def get_linear_leaky_sequential_with_inserted_layer(layers: List[int], # List of fully connected layer sizes
                                                        layer_to_insert: nn.Module, 
                                                        layer_to_insert_index: int,
                                                        layer_to_insert_output_size: int,
                                                        dropout=0.0,
                                                        norm_type="none"):
        """Get sequential fully connected layers, with a layer inserted at a specific index
        """
        if layer_to_insert_index < 0 or layer_to_insert_index > len(layers) - 1:
            raise ValueError(f"layer_to_insert_index must be between 1 and {len(layers) - 1}")

        layers.insert(layer_to_insert_index + 1, layer_to_insert_output_size)

        for i, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            if i == layer_to_insert_index:
                yield layer_to_insert
                continue

            yield nn.Linear(in_size, out_size)
            if norm_type == "batch_norm":
                yield ModelUtils.PermutedBatchNorm(out_size)
            elif norm_type == "layer_norm":
                yield nn.LayerNorm(out_size)
            elif norm_type == "layer_norm_no_elementwise_affine":
                yield nn.LayerNorm(out_size, elementwise_affine=False)
            elif norm_type != "none":
                raise ValueError(f"Unknown norm type {norm_type}")
            
            if dropout > 0.0: 
                # We put relu and dropout in a sequential, as this makes it easier to potentially use pretrained networks 
                # with or without dropout, for a new network with or without dropout. 
                # As both of these layers don't have parameters, but if we put dropout in the outer sequential, 
                # it shifts the indices of the other layers, which means loading state dicts becomes a pain as you now have to 
                # shift all of the indices of the layers after the dropout layer. 
                # E.g: 
                # [0] Linear
                # [1] Leaky
                # [2] Linear
                # If we do
                # [0] Linear
                # [1] Leaky
                # [2] Dropout
                # [3] Linear  <---- Note how the index is now 3, not 2
                # We need to shift the indices of linear. 
                # If we instead do
                # [0] Linear
                # [1] Sequential(Leaky, Dropout)
                # [2] Linear  <---- Index is the same 
                yield nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout))
            else:
                yield nn.LeakyReLU()




if __name__ == "__main__":
    # print(list(ModelUtils.get_linear_leaky_sequential([30, 20, 80])), False)
    # print(list(ModelUtils.get_linear_leaky_sequential([30, 20, 80], True)))

    class TestModule(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.model = torch.nn.LSTM(input_size, output_size)
            self.output_size = output_size
            self.input_size = input_size

        def get_output_size(self):
            return self.output_size
        
    output_size = 100

    model = TestModule(30, output_size)
    print(nn.Sequential(*list(ModelUtils.get_linear_leaky_sequential_with_inserted_layer([30, 20, 80], model, 0, output_size, True))))
    model = TestModule(20, output_size)
    print(nn.Sequential(*list(ModelUtils.get_linear_leaky_sequential_with_inserted_layer([30, 20, 80], model, 1, output_size, True))))
    model = TestModule(80, output_size)
    print(nn.Sequential(*list(ModelUtils.get_linear_leaky_sequential_with_inserted_layer([30, 20, 80], model, 2, output_size, True))))