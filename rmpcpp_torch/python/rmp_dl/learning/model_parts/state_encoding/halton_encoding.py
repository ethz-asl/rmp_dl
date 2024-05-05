

from typing import Dict
import numpy as np
from rmp_dl.util.halton_sequence import HaltonUtils
import torch
import torch.nn as nn

class Halton2dEncoder(nn.Module):
    def __init__(self, 
                 n: int, 
                 method: str,
                 method_parameters: Dict[str, Dict] = {"max": {"k": 1}}
                 ):
        """Encode a direction vector into a halton sequence ray representation. 
        This is a sparse representation of the direction vector, where the rays are
        sampled from a 2d halton sequence (base 2 and 3). 
        Method is the way to convert. Options:
        max: put a single ray that is closest to the direction at 1, and all others at 0
        """
        super().__init__()
        self.n = n
        self.method = method
        self.method_parameters = method_parameters
        
        # These are the unit vector endpoints of the halton sequence rays. Precompute them     
        self.endpoints = torch.from_numpy(HaltonUtils.get_ray_endpoints_from_halton_distances(np.ones(self.n))).to(torch.float32).to(torch.device("cuda"))
        
    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Encodes the given direction into the ray space
        Multiple directions can be given at once, so the size of dimension should be (batch_size, 3, k),
        where n is the number of directions. So e.g. for a position vector and velocity vector, we expect an input
        of size (batch_size, 3, 2)

        """
        # Assert that the input is of the correct size
        
        # For if we are not given multiple directions
        squeeze = False
        if len(directions.shape) == 2: 
            directions = directions.unsqueeze(-1)
            squeeze = True
        assert directions.shape[1] == 3

        if self.method == "max":
            result = self.forward_closest(directions, **self.method_parameters[self.method])
            return result.squeeze(-1) if squeeze else result
        if self.method == "dot":
            result = self.forward_dot(directions, **self.method_parameters[self.method])
            return result.squeeze(-1) if squeeze else result
        else:
            raise ValueError(f"Unknown method {self.method}")


    def forward_closest(self, directions: torch.Tensor, k=1) -> torch.Tensor:
        # We scale the direction to a length of 1, and then take the dot product with the endpoints
        # This gives us the projection of the direction to the ray
        # We then take the largest k projections, and set those to 1, and all others to 0

        distances = self.endpoint_direction_dot(directions, self.endpoints)

        largest_indices = torch.topk(distances, k=k, dim=1)[1]

        # Create a tensor of size (batch_size, n, k) where the largest indices are set to 1, and all others to 0
        rays = torch.zeros_like(distances)
        rays = rays.scatter(1, largest_indices, torch.ones_like(rays).to(torch.float32))

        return rays

    @staticmethod
    def endpoint_direction_dot(directions, endpoints):
        # Normalize the directions
        directions = directions / torch.norm(directions, dim=1, keepdim=True)
        # Take the dot product with the endpoints
        # k is the number of directions
        # b is the batch size
        # n is the number of rays in the halton sequence
        # d is the dimensionality (= 3)
        # So if we have a batch of 8, k directions and n rays, we have an output of (8, n, k), 
        # where each element is the projection of the direction onto the ray for that batch, ray and direction
        distances = torch.einsum("nd,bdk->bnk", endpoints, directions)
        return distances


    def forward_dot(self, directions: torch.Tensor) -> torch.Tensor:
        # We scale the direction to a length of 1, and then take the dot product with the endpoints
        # This gives us the projection of the direction to the ray
        # This is a measure of how close the direction is to this specific ray
        
        # Dot product is [-1, 1], we normalize to [0, 1]
        return self.endpoint_direction_dot(directions, self.endpoints) / 2 + 0.5


if __name__ == "__main__":
    # Tests
    encoder = Halton2dEncoder(16, "max")
    directions = torch.tensor([[[1, 0], [0, 1], [0, 0]]], dtype=torch.float32).to(torch.device("cuda"))
    rays = encoder(directions)

    print(rays)
    print(rays.shape)
    print(torch.sum(rays, dim=1))
    print(torch.sum(rays, dim=2))

    from rmp_dl.vis3d.vis3d import Plot3D
    from rmp_dl.vis3d.utils import Open3dUtils
    p = Plot3D()
    
    
    encoder = Halton2dEncoder(1024, "max", {"max": {"k": 100}})
    direction = torch.tensor([[[1], [1], [0]]], dtype=torch.float32).to(torch.device("cuda"))
    rays = encoder(direction).squeeze().cpu().numpy()

    ray_geom = Open3dUtils.get_rays_geometry(rays, np.array([0, 0, 0]))
    p.add_geometry(ray_geom)

    p.add_sphere(np.array([0, 0, 0]), [0, 1, 0])
    p.add_sphere(direction.squeeze().cpu().numpy(), [1, 0, 0])
    p.vis.run()



