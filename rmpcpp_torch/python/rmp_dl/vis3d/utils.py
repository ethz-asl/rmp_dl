import colorsys
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np

import open3d as o3d
from rmp_dl.util.halton_sequence import HaltonUtils
from rmp_dl.vis3d.line_mesh import LineMesh

class Open3dUtils:
    @staticmethod
    def intensities_to_rgb_with_mpl(intensities, colormap='viridis'):
        """
        Should start using this function instead of the one below
        Converts an array of intensity values (ranging from 0 to 1) to RGB values using a specified Matplotlib colormap.

        Parameters:
        - intensities (np.array): An array of intensity values ranging from 0 to 1.
        - colormap (str): The name of the Matplotlib colormap to use.

        Returns:
        - np.array: An array of RGB values corresponding to the input intensities.
        """
        # Load the colormap
        cmap = plt.get_cmap(colormap)
        
        # Normalize intensities to ensure they are within the valid range
        intensities = np.clip(intensities, 0, 1)
        
        # Convert intensities to RGB using the colormap
        rgb_values = cmap(intensities)[:,:3]  # Slice to exclude the alpha channel
        
        return rgb_values

    @staticmethod
    def get_RGB(norms, maxnorm=None, minnorm=0):
        # Minnorm has default 0 for the velocity, where we want to start at 0
        # 
        # Can also pass in vectors, and we calculate the nor
        if len(norms.shape) == 2 and norms.shape[0] != 1:
            norms = np.linalg.norm(norms, axis=1)
        
        if maxnorm is None:
            maxnorm = max(norms)

        if minnorm is None:
            minnorm = min(norms)

        # First do in HLS, to have constant saturation and lightness
        start = np.array([2/3, 0.5, 1])
        end = np.array([0, 0.5, 1])
        
        mapped = [((x - minnorm) / (maxnorm - minnorm)) * (end - start) + start for x in norms] # type: ignore

        return np.array([colorsys.hls_to_rgb(*x) for x in mapped]) 


    @staticmethod
    def get_trajectory_geometry(positions: List[np.ndarray], 
                                velocities: Optional[List[np.ndarray]] = None, 
                                world_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None, 
                                size=None,
                                color=None, 
                                lines_between_points=False):
        positions = np.array(positions); 
        if velocities is not None:
            velocities = np.array(velocities)
        
        if world_limits is not None:
            indices = np.all((positions >= world_limits[0]) & (positions <= world_limits[1]), axis=1)
            positions = positions[indices]
            if velocities is not None:
                velocities = velocities[indices]
        
        if isinstance(color, list):
            # Color by given color
            c = [color for _ in range(len(positions))]
        elif isinstance(color, np.ndarray):
            if len(color) != len(positions):
                raise RuntimeError("Color list should be same length as positions")
            c = color
        elif color is None: 
            if velocities is None:
                raise RuntimeError("No color given, and no velocities given to color by")
            # Color by velocity
            c = Open3dUtils.get_RGB(velocities)
        else:
            raise RuntimeError("Color should be a single color, list of colors, or None")

        lines_geom = []
        if lines_between_points:
            # Open3d can create lines from a set of points, and indices between these points
            # The indices are pairs of points, so we need to create a list of pairs of indices
            # The first index is always the current position, which we put at index 0
            points = np.array(positions)
            indices = np.stack([np.arange(len(points) - 1), np.arange(1, len(points))], axis=1)

            lines_geom = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(indices)
            )

            lines_geom.colors = o3d.utility.Vector3dVector(np.array(c[:-1]))
            lines_geom = [lines_geom]

        

        if size is not None:
            # This is redefined from Plot3d.get_sphere_geometry (static)
            # Cant import here due to circular import. Should clean this up one day, but don't care right now
            def get_sphere_geometry(location, color, radius=0.05):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(location)
                sphere.paint_uniform_color(color)
                return sphere
            geoms = []
            for pos, colour in zip(positions, c):
                pcd = get_sphere_geometry(pos, color=np.array(colour), radius=size)
                geoms.append(pcd)

            return geoms + lines_geom
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(np.array(c))
        return [pcd] + lines_geom

    @staticmethod
    def get_rays_geometry(rays: np.ndarray, pos: np.ndarray, color: Optional[List[float]]=None, maxnorm=None):
        """Get geometry of rays at position. If no color is given, they are colour coded according to distance

        Args:
            rays (np.ndarray): Rays
            pos (np.ndarrayu): Central position of rays
            color (List[float], optional): RGB color [0-1]. Defaults to None.
        """
        rel_endpoints = HaltonUtils.get_ray_endpoints_from_halton_distances(rays)
        endpoints = rel_endpoints + pos

        # Open3d can create lines from a set of points, and indices between these points
        # The indices are pairs of points, so we need to create a list of pairs of indices
        # The first index is always the current position, which we put at index 0
        start_indices = np.zeros(len(endpoints), dtype=np.int32)
        end_indices = np.arange(1, len(endpoints) + 1, dtype=np.int32)

        indices = np.stack([start_indices, end_indices], axis=1)

        points = np.insert(endpoints, 0, pos, axis=0)

        lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(indices)
        )

        if color is not None:
            lines.paint_uniform_color(color)
        else:
            lines.colors = o3d.utility.Vector3dVector(Open3dUtils.get_RGB(rel_endpoints, maxnorm=maxnorm))

        return lines
        


    @staticmethod
    def get_rays_geometry_mesh(rays: np.ndarray, pos: np.ndarray, color: Optional[List[float]]=None, maxnorm=None, radius=0.03):
        """Get geometry of rays at position. If no color is given, they are colour coded according to distance
        This function uses meshes, which is slower than the lineset version. But we can adjust thickness
        See:
        https://github.com/isl-org/Open3D/pull/738
        https://github.com/isl-org/Open3D/issues/1480
        
        Args:
            rays (np.ndarray): Rays
            pos (np.ndarrayu): Central position of rays
            color (List[float], optional): RGB color [0-1]. Defaults to None.
        """
        rel_endpoints = HaltonUtils.get_ray_endpoints_from_halton_distances(rays)
        endpoints = rel_endpoints + pos

        # Open3d can create lines from a set of points, and indices between these points
        # The indices are pairs of points, so we need to create a list of pairs of indices
        # The first index is always the current position, which we put at index 0
        start_indices = np.zeros(len(endpoints), dtype=np.int32)
        end_indices = np.arange(1, len(endpoints) + 1, dtype=np.int32)

        indices = np.stack([start_indices, end_indices], axis=1)
        points = np.insert(endpoints, 0, pos, axis=0)


        if color is None:
            color = o3d.utility.Vector3dVector(Open3dUtils.get_RGB(rel_endpoints, maxnorm=maxnorm))

        line_mesh = LineMesh(points, indices, color, radius=radius)
        return line_mesh.cylinder_segments
