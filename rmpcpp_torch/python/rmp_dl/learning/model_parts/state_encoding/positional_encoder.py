import math
import torch

import pytorch_lightning as pl

class PositionalEncoder(pl.LightningModule):
    """Positional encoder as described in the NERF paper https://arxiv.org/pdf/2003.08934.pdf
    """
    def __init__(self, L):
        super().__init__()
        self.L = L
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x is a tensor of shape (batch_size, n)

        # Initialize an empty tensor to hold the results
        results = torch.empty((*x.shape, 2*self.L), device=x.device, dtype=x.dtype)  # (batch_size, n, 2L)

        # Calculate sin and cos for each level
        for l in range(self.L):
            results[..., 2*l] = torch.sin(2 ** l * math.pi * x)
            results[..., 2*l + 1] = torch.cos(2 ** l * math.pi * x)

        results = results.reshape((x.shape[0], -1))  # (batch_size, 2Ln)

        return results


if __name__ == "__main__":
    L = 2
    batch_size = 3
    n = 4

    x = torch.rand((batch_size, n))

    encoder = PositionalEncoder(L)

    y = encoder(x)

    print(y.shape)
    print(x)
    print(y)    