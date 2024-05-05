import pickle
from typing import List, Optional, Tuple
from matplotlib import cm

from PIL import Image

import subprocess

import numpy as np
import open3d as o3d

# Lots of stuff comes from the rmp_dl.vis3d package, but we don't want to import that here,
# to keep this package independent of rmp_dl for easy docker usage
class Open3dRender:
    def __init__(self, directory):
        self.directory = directory
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=True)

        world_mesh = self._get_world_geometry(directory)
        world_mesh.compute_vertex_normals()
        self.vis.add_geometry(world_mesh)
        trajectories: List[np.ndarray] = self._get_trajectories(directory)

        bbox = world_mesh.get_axis_aligned_bounding_box()
        bbox.color = [1, 0, 0]
        self.vis.add_geometry(bbox)

        goal = self._get_goal(directory)
        self.vis.add_geometry(Open3dRender.get_sphere_geometry(goal, [0, 1, 0]))

        color_map = cm.get_cmap('viridis', len(trajectories))
        for i, trajectory in enumerate(trajectories):
            self.vis.add_geometry(Open3dRender.get_trajectory_geometry(trajectory, color=color_map(i)[:3]))

    @staticmethod
    def _get_goal(dir):
        return pickle.load(open(f"{dir}/goal.pkl", "rb"))

    @staticmethod
    def _get_world_geometry(dir):
        return o3d.io.read_triangle_mesh(f"{dir}/world.ply")
        
    @staticmethod
    def _get_trajectories(dir) -> List[np.ndarray]:
        return pickle.load(open(f"{dir}/trajectories.pkl", "rb"))

    @staticmethod
    def _get_camera_views(dir) -> List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        return pickle.load(open(f"{dir}/camera_views.pkl", "rb"))

    def save_images(self):
        camera_views = self._get_camera_views(self.directory)

        for name, (pos, look_at, up) in camera_views:
            print("Saving image")
            print(name)
            print(pos)
            print(look_at)
            print(up)
            vc = self.vis.get_view_control()
            vc.set_lookat(look_at)
            vc.set_front(pos)
            vc.set_up(up)
            
            self._save_image(f"{self.directory}/output/{name}.jpeg")
            
    def _save_image(self, filename):  
        self.vis.poll_events()
        self.vis.update_renderer()

        self.vis.capture_screen_image(filename, do_render=True)
        self.optimize_image(filename)

    @staticmethod
    def optimize_image(filename):
        img = Image.open(filename)
        if img.size > (1280, 720):
            img = img.resize((1280, 720), Image.LANCZOS) # Some sort of antialiasing filter

        img.save(filename, optimize=True, quality=85)
    
    @staticmethod
    def get_sphere_geometry(location, color, radius=0.05):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(location)
        sphere.paint_uniform_color(color)
        return sphere

    @staticmethod
    def get_trajectory_geometry(positions: np.ndarray, color):
        # Color by given color
        c = [color for _ in range(len(positions))]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(np.array(c))

        return pcd

if __name__ == "__main__":
    vis = Open3dRender("/opt/data")
    vis.save_images()

    
