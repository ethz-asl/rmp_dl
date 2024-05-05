import abc
import numpy as np
from rmp_dl.util.halton_sequence import HaltonUtils
import torch
import torch.nn as nn


class DecoderMethod(abc.ABC, nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        
        # These are the unit vector endpoints of the halton sequence rays. Precompute them     
        points = HaltonUtils.get_ray_endpoints_from_halton_distances(np.ones(self.n))
        self.endpoints = torch.from_numpy(points)
        self.endpoints = self.endpoints.to(torch.float32)
        self.endpoints = self.endpoints.to(torch.device("cuda"))

    @abc.abstractmethod
    def forward(self, rays: torch.Tensor) -> torch.Tensor: ...

    
class MaxSumDecoder(DecoderMethod):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

        self.softmax = nn.Softmax(dim=1)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Get the average of the k largest values for each batch
        """
        rays = self.softmax(rays)

        # Get the values and indices of the maximum k values for each batch
        # Both are size (batch_size, k)
        max_values, max_indices = torch.topk(rays, k=self.k, dim=1)

        # Get the endpoints for each ray
        endpoints = torch.index_select(self.endpoints, dim=0, index=max_indices.reshape(-1)).reshape(-1, self.k, 3) # (batch_size, k, 3)

        # scale the endpoints by the ray values
        endpoints = endpoints * max_values.unsqueeze(-1)  # (batch_size, k, 3)

        # Get the average of the endpoints
        # This is of size (batch_size, 3)
        return nn.functional.normalize(torch.mean(endpoints, dim=1))

class Halton2dDecoder(nn.Module):
    def __init__(self, 
                 n: int, 
                 method: str,
                 method_parameters: dict,
                 ):
        super().__init__()
        self.method = method
        self.method_parameters = method_parameters
        self.n = n 
        
        self.forward_method: DecoderMethod = self._resolve_method(method, method_parameters)
        
    def _resolve_method(self, method: str, method_parameters: dict) -> DecoderMethod:
        if method == "max":
            return MaxSumDecoder(n=self.n, **method_parameters)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        return self.forward_method(rays)


class Halton2dDecoderFactory:
    @staticmethod
    def resolve_decoder(method: str, n: int):
        if method == "max_decoder":
            return Halton2dDecoder(n=n, method="max", method_parameters={"k": 1})
        elif method == "max_sum10_decoder":
            return Halton2dDecoder(n=n, method="max", method_parameters={"k": 10})
        elif method == "max_sum50_decoder":
            return Halton2dDecoder(n=n, method="max", method_parameters={"k": 50})
        elif method == "max_sum100_decoder":
            return Halton2dDecoder(n=n, method="max", method_parameters={"k": 100})
        elif method == "max_sum512_decoder":
            return Halton2dDecoder(n=n, method="max", method_parameters={"k": 512})
        elif method == "max_sum1024_decoder":
            return Halton2dDecoder(n=n, method="max", method_parameters={"k": 1024})
        else:
            raise NotImplementedError(f"Method {method} not implemented")
    @staticmethod
    def max_sum50_decoder(n: int) -> Halton2dDecoder:
        return Halton2dDecoder(n=n, method="max", method_parameters={"k": 50})


if __name__ == "__main__":
    x = torch.randn(8, 1024).to(torch.device("cuda"))
    encoder = Halton2dDecoder(1024, "max")
    y = encoder(x)
    pass
