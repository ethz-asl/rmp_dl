import numpy as np
from pathlib import Path
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from rmp_dl.learning.data.pipeline.nodes.samplers.position_sampler import PositionSampler
from rmp_dl.learning.data.utils.location_sampler import UniformWorldLimitsSampler
from rmp_dl.planner.planner_params import RayObserverParameters

from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.util.voxel_conversions import VoxelConversions
from rmp_dl.vis3d.arrow import get_arrow_geometry
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase



class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val, world_limits):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    linear_width = np.linalg.norm(world_limits[1] - world_limits[0]) / 2
    self.norm = mpl.colors.AsinhNorm(linear_width=linear_width, vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


class GeodesicVis:
    def __init__(self, plot3d: Plot3D, distancefield_gen: DistanceFieldGen, 
                 num_arrows=9e3, dx=(0.0, 0.0), dy=(0.0, 0.0), dz=(0.0, 0.0)):
        self.arrows = None
        inflation = distancefield_gen.inflation
        self.distancefield_gen = distancefield_gen
        gradient_interpolator = self.distancefield_gen.get_gradient_interpolate
        distancefield_interpolator = self.distancefield_gen.get_distancefield_interpolate
        esdf_interpolator = self.distancefield_gen.get_esdf_interpolate
        world_limits = self.distancefield_gen.world_limits
        voxel_size = self.distancefield_gen.voxel_size

        self.plot3d = plot3d
        self.esdf_interpolator = esdf_interpolator
        self.world_limits = world_limits

        v2 = voxel_size / 2
        minx = world_limits[0][0] + v2 + dx[0]; maxx = world_limits[1][0] - v2 - dx[1]
        miny = world_limits[0][1] + v2 + dy[0]; maxy = world_limits[1][1] - v2 - dy[1]
        minz = world_limits[0][2] + v2 + dz[0]; maxz = world_limits[1][2] - v2 - dz[1]

        sx = maxx - minx
        sy = maxy - miny
        sz = maxz - minz 

        volume = sx * sy * sz 
        v2 = voxel_size / 2

        arrow_density = np.cbrt(num_arrows / volume)
        x = np.arange(minx + v2, maxx - v2, 1 / arrow_density)
        y = np.arange(miny + v2, maxy - v2, 1 / arrow_density)
        z = np.arange(minz + v2, maxz - v2, 1 / arrow_density)

        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        self.coordinates = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

        self.geodesic = np.array([distancefield_interpolator(c) if esdf_interpolator(c) > inflation else np.array([0]) for c in self.coordinates])
        self.gradient = np.squeeze(np.array([gradient_interpolator(c) for c in self.coordinates]))

        self.converter: VoxelConversions = VoxelConversions(world_limits=world_limits, voxel_size=voxel_size)
        self.colorHelper = MplColorHelper("plasma", self.geodesic.min(), self.geodesic.max(), world_limits=world_limits)

    def _generate_geodesic(self):
        norms = np.linalg.norm(self.gradient, axis=1)
        norms[norms == 0] = float("NaN")  # to avoid divide by 0 below
        self.gradient /= np.repeat(norms[..., np.newaxis], 3, axis=1)

        self.colors = list(map(self.colorHelper.get_rgb, self.geodesic.flatten()))
        
        zipped = zip(self.coordinates, self.gradient, self.colors, self.geodesic)
        filtered = list(filter(lambda x: x[-1] != 0, zipped))  # When the geodesic is (very close to 0), we are inside an obstacle

        world_length = np.sum(self.world_limits[1] - self.world_limits[0])
        arrow_size = 0.175 * world_length / (10.4 * 3)  # Some heuristic that looks okay

        self.arrows = [get_arrow_geometry(coordinate, grad, arrow_size) for (coordinate, grad, _, _) in filtered]
        colors = [color for (_, _, color, _) in filtered]

        self.pc = o3d.geometry.PointCloud()
        self.pc.points = o3d.utility.Vector3dVector(np.array([x[0] for x in filtered]))
        self.pc.colors = o3d.utility.Vector3dVector(np.array(colors)[:, :3])

        for i, arrow in enumerate(self.arrows):
            arrow.paint_uniform_color(colors[i][:3])


    def show_geodesic(self):
        if self.arrows is None: 
           self._generate_geodesic()
        for arrow in self.arrows if self.arrows else []:
            self.plot3d.add_geometry(arrow)
        self.plot3d.add_geometry(self.pc)

    def hide_geodesic(self):
      for arrow in self.arrows if self.arrows else []:
            self.plot3d.remove_geometry(arrow)
      self.plot3d.remove_geometry(self.pc)

class DenseSampledVis:
    def __init__(self, plot3d: Plot3D, worldgen: WorldgenBase):
        self.plot3d = plot3d
        self.worldgen = worldgen

    def plot_locations(self, num=1024, scale=0.2):
        for observation in self.get_observations(self.worldgen, num):
          arrow = get_arrow_geometry(observation["state"]["pos"], observation["expert_policy"]["geodesic"], scale=scale)
          arrow.paint_uniform_color([1, 0, 0])
          self.plot3d.add_geometry(arrow)

    @staticmethod
    def get_observations(worldgen, num):
      sampler = UniformWorldLimitsSampler(0)
      position_sampling_function = PositionSampler.PositionSamplingFunction(num, RayObserverParameters.from_yaml_general_config(), sampler)

      # The sampling function expects a callable (comes from some multiprocessing things in the datapipeline that 
      # it needs a callable)
      world_constructor = lambda: worldgen  
      data, _ = position_sampling_function.__call__(world_constructor)

      for observation in data:
        yield observation

      
      
          