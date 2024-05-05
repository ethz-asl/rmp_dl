
from functools import lru_cache
import numpy as np


class HaltonUtils:

    @staticmethod
    def get_angles(u, v):
        phi = np.arccos(1 - 2 * u)
        theta = 2 * np.pi * v
        return phi, theta


    @staticmethod 
    @lru_cache
    def get_u_v(length):
        u = np.array([HaltonUtils.get_halton_seq(i, 2) for i in range(length)])
        v = np.array([HaltonUtils.get_halton_seq(i, 3) for i in range(length)])
        return u, v


    @staticmethod
    def get_ray_endpoints_from_halton_distances(rays: np.ndarray):
        """Gets the endpoints from rays that used the halton sequence to generate the directions
        Assumes origin is (0, 0, 0)
        """
        # The cpp code uses halton sequences with base 2 for <u, phi>, and base 3 for <v, theta>
        u, v = HaltonUtils.get_u_v(len(rays))

        phi, theta = HaltonUtils.get_angles(u, v)

        # Convert to cartesian unit vector
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Scale by ray length
        x = x * rays
        y = y * rays
        z = z * rays

        return np.stack([x, y, z], axis=1)

    @staticmethod
    def get_halton_seq(index, base):
        """Get the $base halton sequence at $index
        https://en.wikipedia.org/wiki/Halton_sequence
        """
        f = 1.0
        r = 0.0
        while index > 0:
            f = f / base
            r = r + f * (index % base)
            index = index // base
        return r
