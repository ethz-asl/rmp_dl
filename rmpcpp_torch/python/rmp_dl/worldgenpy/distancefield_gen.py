import time
from typing import Optional, Tuple
from rmp_dl.learning.data.utils.location_sampler import UniformWorldLimitsSampler
import skfmm
import numpy as np

import scipy
import scipy.interpolate

class DistanceFieldGen:
    def __init__(self, esdf: np.ndarray, goal: np.ndarray, 
                 world_limits, voxel_size: float, 
                 df: Optional[np.ndarray]=None, 
                 inflation: float=0.0):
        """Class to generate a distance field from an ESDF.

        Args:
            df (np.ndarray, optional): If we have a cached distancefield, it can be passed directly and it will not be regenerated. Defaults to None.
        """
        self.world_limits = world_limits
        self.voxel_size = voxel_size
        self.inflation = inflation

        upsample_factor = 2
        esdf, locations, X, Y, Z = DistanceFieldGen._get_upsampled_esdf_and_locations(esdf, voxel_size, world_limits, upsample_factor)

        if df is None: 
            self._distancefield = DistanceFieldGen._generate_distancefield(voxel_size / upsample_factor, goal, locations, esdf, inflation)
        else: 
            self._distancefield = df
        
        self._distancefield_interpolator = scipy.interpolate.RegularGridInterpolator((X, Y, Z), self._distancefield, method='linear')
        if df is not None:
            if self.get_distancefield_interpolate(goal) > voxel_size:
                print("WARNING: You are using a cached distancefield, but the distance to the goal is greater than the voxel size.  \
                      This means that the cached distancefield is not accurate, so something is wrong with caching. We will regenerate the distancefield.")
                self._distancefield = DistanceFieldGen._generate_distancefield(voxel_size / upsample_factor, goal, locations, esdf)


        self._gradient = -np.array(np.gradient(self._distancefield, voxel_size / upsample_factor))
        self._esdf_interpolator = scipy.interpolate.RegularGridInterpolator((X, Y, Z), esdf, method='linear', bounds_error=False, fill_value=None)
        self._gradient_interpolator = scipy.interpolate.RegularGridInterpolator((X, Y, Z), self._gradient.transpose([1, 2, 3, 0]), 
                                                                                method='linear', bounds_error=False, fill_value=None)


    @staticmethod
    def _get_upsampled_esdf_and_locations(esdf, voxel_size, world_limits, upsample_factor):
        v2 = voxel_size / 2
        minx = world_limits[0][0]; maxx = world_limits[1][0]
        miny = world_limits[0][1]; maxy = world_limits[1][1]
        minz = world_limits[0][2]; maxz = world_limits[1][2]

        # Our datapoints lie on the centre of voxels. 
        x = np.arange(np.floor(minx / voxel_size) * voxel_size + v2, np.ceil(maxx / voxel_size) * voxel_size + v2, voxel_size)
        y = np.arange(np.floor(miny / voxel_size) * voxel_size + v2, np.ceil(maxy / voxel_size) * voxel_size + v2, voxel_size)
        z = np.arange(np.floor(minz / voxel_size) * voxel_size + v2, np.ceil(maxz / voxel_size) * voxel_size + v2, voxel_size)

        sx = maxx - minx
        sy = maxy - miny
        sz = maxz - minz

        X = np.linspace(world_limits[0][0] + voxel_size, world_limits[1][0] - v2, round(sx / voxel_size * upsample_factor))
        Y = np.linspace(world_limits[0][1] + voxel_size, world_limits[1][1] - v2, round(sy / voxel_size * upsample_factor))
        Z = np.linspace(world_limits[0][2] + voxel_size, world_limits[1][2] - v2, round(sz / voxel_size * upsample_factor))
        locations = np.stack(np.meshgrid(X, Y, Z, indexing='ij')).transpose([1, 2, 3, 0]) # (dimx, dimy, dimz, 3)

        interp = scipy.interpolate.RegularGridInterpolator((x, y, z), esdf, method='linear', bounds_error=False, fill_value=None)
        esdf = interp(locations.reshape(-1, 3)).reshape(locations.shape[:-1])

        return esdf, locations, X, Y, Z

    @staticmethod
    def _generate_distancefield(voxel_size, goal, locations, esdf, inflation):
        # (dim, dim, dim). Zero contour around the goal a distance of 2x voxel_size away. 
        # Making this too small results in having no zero level set
        # Also with inflating the obstacles, we need to make sure that the zero level set is not too close to the obstacles
        phi = np.linalg.norm(goal - locations, axis=3) - 2 * voxel_size 

        speed = np.zeros_like(esdf)
        speed[esdf > inflation] = 1  
        
        distancefield: np.ndarray = skfmm.travel_time(
            phi=phi, speed=speed, dx=voxel_size, order=1  # order 1 seems to handle sharp corners around obstacles better
        )

        # We do nearest neighbor interpolation for (inflated) obstacles
        distancefield[esdf <= inflation] = float('NaN')
        mask = np.where(~np.isnan(distancefield))
        nan_interpolation = scipy.interpolate.NearestNDInterpolator(np.transpose(mask), distancefield[mask])
        filled_data = nan_interpolation(*np.indices(distancefield.shape))  # It wants interpolation dimension in the last dim

        inflation_region = np.logical_and(esdf >= 0, esdf <= inflation)
        filled_data[inflation_region] += inflation - esdf[inflation_region]

        return filled_data


    def get_esdf_interpolate(self, loc: np.ndarray) -> float:
        return self._esdf_interpolator(loc)

    def get_gradient_interpolate(self, loc: np.ndarray) -> np.ndarray:
        return self._gradient_interpolator(loc)

    def get_distancefield_interpolate(self, loc: np.ndarray) -> float:
        return self._distancefield_interpolator(loc)

    def get(self):
        return self._distancefield


    def fraction_valid_and_reachable(self, N=100):
        """We sample N points inside the free area (outside inflated obstacles), and check which fraction of them are reachable from the goal.

        Args:
            N (int, optional): _description_. Defaults to 100.
        """
        count = 0
        valid_count = 0
        reachable_count = 0
        limits = (self.world_limits[0] + self.voxel_size, self.world_limits[1] - self.voxel_size)
        for loc in UniformWorldLimitsSampler(seed=0).sample(limits=self.world_limits):
            count += 1
            if self.get_esdf_interpolate(loc) <= self.inflation:
                continue
            valid_count += 1

            if self.get_distancefield_interpolate(loc) != 0:
                reachable_count += 1

            if valid_count >= N:
                break
        
        return valid_count / count, reachable_count / valid_count

