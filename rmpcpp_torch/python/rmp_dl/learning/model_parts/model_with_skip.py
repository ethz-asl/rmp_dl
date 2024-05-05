import torch

    
class RecurrentModelWithSkipAdd(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, hiddens):
        identity = x
        x, hiddens = self.model(x, hiddens)
        x = x + identity
        return x, hiddens