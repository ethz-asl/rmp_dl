import collections
from typing import Callable, List, Optional
import open3d as o3d

import numpy as np
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

class KeyModifier:
    NONE = 0
    SHIFT = 1
    CTRL = 2
    ALT = 4

class Plot3D():
    ARROW_RIGHT = 262
    ARROW_LEFT = 263
    ARROW_DOWN = 264
    ARROW_UP = 265
    COMMA = 44
    
    SPACEBAR = 32

    def __init__(self):
        # Create a window.
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.window = self.vis.create_window(visible=True)

        self.callables = collections.defaultdict(list)
 
    @staticmethod 
    def get_world_geometry(worldgen):
        path = worldgen.export_to_ply()
        
        # Load the mesh.
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        return mesh


    def plot_world(self, worldgen: WorldgenBase, do_normal_coloring: bool=False, draw_bbox: bool=True):
        mesh = Plot3D.get_world_geometry(worldgen)

        self.add_geometry(mesh)
        # Color the mesh by normals.
        if do_normal_coloring:
            self.vis.get_render_option().mesh_color_option = \
                o3d.visualization.MeshColorOption.Normal
        
        if draw_bbox:
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox.color = [1, 0, 0]
            self.add_geometry(bbox)
        
            self.create_origin_boxes(worldgen.get_voxel_size())

            limits = worldgen.get_world_limits()

            bbox2 = o3d.geometry.AxisAlignedBoundingBox(*limits)
            bbox2.color = [0, 0, 1]

            self.add_geometry(bbox2)
            
        
        return mesh  # We return the mesh geometry so it can also be removed
    
    def plot_start_goal(self, worldgen, start=True, goal=True):
        if start:
            self.add_sphere(worldgen.get_start(), color=np.array([0, 0, 1]), radius=0.05)
        if goal:
            self.add_sphere(worldgen.get_goal(), color=np.array([0, 1, 0]), radius=0.3)

    def create_origin_boxes(self, voxel_size):
        # First we put a box at the origin
        box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
        box.paint_uniform_color([0, 0, 0])
        self.add_geometry(box)
        
        box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
        box.translate([voxel_size, 0, 0])
        box.paint_uniform_color([1, 0, 0])
        self.add_geometry(box)

        box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
        box.translate([0, voxel_size, 0])
        box.paint_uniform_color([0, 1, 0])
        self.add_geometry(box)

        box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
        box.translate([0, 0, voxel_size])
        box.paint_uniform_color([0, 0, 1])
        self.add_geometry(box)


    def plot_voxel_world(self, occupancy_grid, world_limits, voxel_size, voxel_grid_loc, voxel_grid_size):

        # Get the lower bound indices of the grid location
        lower_corner = voxel_grid_loc - voxel_grid_size / 2
        lower_index = np.floor((lower_corner - world_limits[0]) / voxel_size).astype(int)

        # Get the upper bound indices of the grid location
        upper_corner = voxel_grid_loc + voxel_grid_size / 2
        upper_index = np.ceil((upper_corner - world_limits[0]) / voxel_size).astype(int)

        grid_size = np.array(occupancy_grid.shape)

        indices = np.indices(upper_index - lower_index, dtype=int) + lower_index[:, np.newaxis, np.newaxis, np.newaxis]
        occupancy_grid_small = occupancy_grid[lower_index[0]:upper_index[0], lower_index[1]:upper_index[1], lower_index[2]:upper_index[2]]
        occupancy_grid_small = occupancy_grid_small[np.newaxis, :]
        
        # We concatenate the indices and the value, and flatten 
        indices_and_values = np.concatenate((indices, occupancy_grid_small)).reshape(4, -1)
        
        # Find the indices where the value of the occupancy grid is <= 0 (= occupied)
        occupied = indices_and_values[:3, indices_and_values[3, :] <= 0].T
        
        # We only want to show the voxels on the edges, so we remove voxels that are surrounded by other occupied voxels
        def filt(index):
            x, y, z = index.astype(int)
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        if i == 0 and j == 0 and k == 0:
                            continue
                        if x + i < 0 or x + i >= grid_size[0] or y + j < 0 or y + j >= grid_size[1] or z + k < 0 or z + k >= grid_size[2]:
                            continue
                        if occupancy_grid[x + i, y + j, z + k] > 0.:
                            return True
            return False

        print(f"Occupied: {len(occupied)}")
        occupied = np.array([occ for occ in occupied if filt(occ)])
        print(f"Occupied edges: {len(occupied)}")

        # Now convert the indices to locations. The mesh used later on positions the corner at (0, 0, 0), so we also need the corner location of the voxel
        occupied *= voxel_size
        occupied += world_limits[0]


        # if len(occupied) > 5000:
            # return
        geometries = []
        for i, occ in enumerate(occupied):
            print(f"{i}/{len(occupied)}\r")

            box = o3d.geometry.TriangleMesh.create_box(voxel_size / 4, voxel_size / 4, voxel_size / 4)
            box.translate(np.array([voxel_size / 8 * 3, voxel_size / 8 * 3, voxel_size / 8 * 3]))
            box.translate(occ)
            box.paint_uniform_color([0.1, 0.1, 0.1])
            
            bb = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size).get_axis_aligned_bounding_box()
            bb.translate(occ)
            bb.color = [1, 0, 0]
            self.add_geometry(box)
            self.add_geometry(bb)
            geometries += [box, bb]
        return geometries

    @staticmethod
    def get_sphere_geometry(location, color, radius=0.05):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(location)
        sphere.paint_uniform_color(color)
        return sphere

    def add_sphere(self, location, color, radius=0.05):
        sphere = Plot3D.get_sphere_geometry(location, color, radius)
        self.add_geometry(sphere)

    def add_geometry(self, geometry):
        self.vis.add_geometry(geometry)

    def remove_geometry(self, geometry):
        self.vis.remove_geometry(geometry)

    def register_callback(self, key: int, callable: Callable[[int, int], None]):
        """Registers a callback for the given key. Keys are within GLFW system
        https://www.glfw.org/docs/latest/group__keys.html

        If called multiple times with the same key, we call all callables
        """
        self.callables[key].append(callable)

        if len(self.callables[key]) == 1:
            # First time we have registered
            def call_all_callbacks(vis, action, mods):
                for callable in self.callables[key]:
                    self._do_callback_with_render_update(vis, action, mods, callable)
            
            self.vis.register_key_action_callback(key, call_all_callbacks)

        
    def _do_callback_with_render_update(self, visualizer, action: int, mods: int, callable: Callable[[int, int], None]):
        """A callback wrapper which calls the given callable with the given arguments, 
        and then updates the renderer. 
        Args:
            action (int): Action of the key: 0 = release, 1 = press, 2 = repeat. 
            mods (int): Modifier on the press: Basically 0 is nothing, anything else is a key (uses glfw key codes). 
                We usually just use 0 or 1 for shift.
            callable(Callable[[int, int], None]): Callable that takes in the action and mods. 

        Returns:
            _type_: _description_
        """
        if action != 1:
            return False    
        
        vc = visualizer.get_view_control()
        params = vc.convert_to_pinhole_camera_parameters()

        callable(action, mods)

        visualizer.get_view_control().convert_from_pinhole_camera_parameters(params)
        self.vis.poll_events()
        self.vis.update_renderer()
        return True


    def run(self):
        self.vis.run()

    def destroy(self):
        self.vis.destroy_window()

if __name__ == "__main__":
    plot = Plot3D()
    plot.plot_world("/home/isar/Documents/Studie/Zurich/RA/projects/rmp_dl_planning/src/asdf.ply", False)
    plot.vis.run()
    
