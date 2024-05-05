import copy
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.planner import PlannerRmp
from rmp_dl.vis3d.geodesic_vis import GeodesicVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.vis3d.vis3d import KeyModifier, Plot3D
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.vis3d.vis3d_stepped import Plot3DStepped
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory

import time

import open3d as o3d

from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

class Planner3dVis:
    def __init__(self, worldgen, 
                 distancefield=False, 
                 distancefield_inflation = 0.0,
                 start=True, goal=True, draw_bbox=True,
                 # Params for distancefield
                 num_arrows=9e3, dx=(0.0, 0.0), dy=(0.0, 0.0), dz=(0.0, 0.0), 
                 initial_idx=0, initial_modifier=0,
                 voxel_grid=False,
                 voxel_grid_loc=np.array([5.0, 5.0, 5.0]),
                 voxel_grid_size=np.array([1.0, 1.0, 1.0])
                ):
        self.plot3d = Plot3DStepped(initial_idx=initial_idx, initial_modifier=initial_modifier)
        self.plot_voxel_grid = voxel_grid
        self.voxel_grid_loc = voxel_grid_loc
        self.voxel_grid_size = voxel_grid_size


        self.worldgen: WorldgenBase = worldgen

        self.params = {}
        self.trajectory_geometry = {}

        self.plot_world(worldgen, start, goal, draw_bbox)

        if distancefield: 
            self.plot_distancefield(num_arrows, dx, dy, dz, distancefield_inflation)
            self._setup_geodesic_hide_callback()

        self._setup_trajectory_hide_callback()

    def plot_world(self, worldgen, start=True, goal=True, draw_bbox=True):
        self.world_mesh = self.plot3d.plot_world(worldgen, draw_bbox=draw_bbox)

        if worldgen.get_world_limits() is not None and self.plot_voxel_grid:
            self.voxel_grid_geometries = self.plot3d.plot_voxel_world(worldgen.get_esdf_as_ndarray(), worldgen.get_world_limits(), worldgen.get_voxel_size(), 
                                         self.voxel_grid_loc, self.voxel_grid_size)
        
        self.plot3d.plot_start_goal(worldgen, start, goal)
        self._setup_world_hide_callback()
        self._setup_voxel_grid_hide_callback()

    @property
    def observations(self):
        return self.plot3d.observations

    def go(self):
        self.plot3d.run()

    def destroy(self):
        self.plot3d.destroy()

    def increment_and_get_step_geometry_idx(self):
        return self.plot3d.increment_and_get_step_geometry_idx()

    def register_step_geometry_change(self, getter):
        self.plot3d.register_step_geometry_change(getter)

    def _setup_world_hide_callback(self):
        def plot_world():
            self.plot3d.add_geometry(self.world_mesh)
        def hide_world():
            self.plot3d.remove_geometry(self.world_mesh)
        
        self.is_world_plotted = True

        def toggle_world_callback(action):
            if action == 0: return
            if self.is_world_plotted:
                hide_world()
            else:
                plot_world()
            self.is_world_plotted = not self.is_world_plotted

        self.plot3d.register_callback(ord("W"), lambda action, mods: toggle_world_callback(action))

    def _setup_geodesic_hide_callback(self):
        def plot_geodesic():
            self.geovis.show_geodesic()
        def hide_geodesic():
            self.geovis.hide_geodesic()
        
        self.is_geodesic_plotted = True

        def toggle_geodesic_callback(action):
            if action == 0: return
            if self.is_geodesic_plotted:
                hide_geodesic()
            else:
                plot_geodesic()
            self.is_geodesic_plotted = not self.is_geodesic_plotted

        self.plot3d.register_callback(ord("G"), lambda action, mods: toggle_geodesic_callback(action))

    def _setup_trajectory_hide_callback(self):
        def plot_trajectory():
            for name, geometry in self.trajectory_geometry.items():
                self.add_geometry(geometry)
        def hide_trajectory():
            for name, geometry in self.trajectory_geometry.items():
                self.remove_geometry(geometry)
        
        self.is_trajectory_plotted = True

        def toggle_trajectory_callback(action):
            if action == 0: return
            if self.is_trajectory_plotted:
                hide_trajectory()
            else:
                plot_trajectory()
            self.is_trajectory_plotted = not self.is_trajectory_plotted

        self.plot3d.register_callback(ord("T"), lambda action, mods: toggle_trajectory_callback(action))

    def _setup_voxel_grid_hide_callback(self):
        def plot_voxel_grid():
            if not hasattr(self, "voxel_grid_geometries"): return
            for geometry in self.voxel_grid_geometries:
                self.add_geometry(geometry)
        def hide_voxel_grid():
            if not hasattr(self, "voxel_grid_geometries"): return
            for geometry in self.voxel_grid_geometries:
                self.remove_geometry(geometry)
        
        self.is_voxel_grid_plotted = True
        def toggle_voxel_grid_callback(action):
            if action == 0: return
            if self.is_voxel_grid_plotted:
                hide_voxel_grid()
            else:
                plot_voxel_grid()
            self.is_voxel_grid_plotted = not self.is_voxel_grid_plotted
        
        self.plot3d.register_callback(ord("P"), lambda action, mods: toggle_voxel_grid_callback(action))

    def add_planner(self, name, planner_create: Callable[..., PlannerRmp], 
                    planner_params: dict,
                    key_replan: List[Tuple[str, Callable[[None], float], float]], 
                    color=Optional[Union[List, Callable]], 
                    trajectory_size=None) -> None:
        """Add a planner to the visualizer. 
        The planner create callable is used to create the planner, passing the planner_params dict as kwargs to the constructor.
        The key_replan list can be used to create parameter change callbacks which trigger a replan, mapped to keys during 3d visualization. 
        Every tuple is of (key, getter, setter, increment). The key is the key to press to trigger the callback, the 
        getter should accept a modifier (e.g. shift, ctrl, see o3d docs for values) as input. 


        Args:
            name (str): Name of the planner. Not very important, just make sure that every call to this function has a unique name.
            planner_create (Callable): Constructor to create the planner. Arguments to the constructor can be supplied inside planner params
            key_replan (List[Tuple[str, str, str, float]]): List of tuples to update increment/decrement the planner parameters and replan. 
                Each entrty is a tuple of (key, setter). 
            color (Optional[Union[List, Callable]], optional): Color of the trajectory. Can be a list of [r, g, b] values, or a callable which takes the observations dict and returns a list of colors. 
                Defaults to None. If None velocity is used (if available)
            trajectory_size: Size of the trajectory. If None (recommended) a pointcloud geometry is used to visualize. If you want to manually set the size, 
                sphere geometries are used which are less performant, so only use this for making figures for a report or something. 
        """
        self.plot3d.current_step_geometry_modifier_steps = 0
        if name in self.params:
            raise RuntimeError("Name is already in use")
        self.params[name] = planner_params
        self.observations[name] = []

        def replan():
            self.observations[name].clear()
            planner: PlannerRmp = planner_create(**self.params[name])
            distancefield = self.worldgen.get_distancefield() if planner.requires_geodesic else None
            esdf = self.worldgen.get_esdf() if planner.requires_esdf else None
            planner.setup(self.worldgen.get_start(), self.worldgen.get_goal(), self.worldgen.get_tsdf(), esdf, distancefield)
            start = time.time()
            while True:
                observations, terminated = planner.step(1, terminate_if_stuck=False)
                self.observations[name].append(observations)
                if terminated:
                    break
            t = time.time() - start
            print(f"Planner {name} took {t} seconds to plan, success: {planner.success()}, integration speed: {planner.get_trajectory_length() / t} hz")
            trajectory = planner.get_trajectory()

            if name in self.trajectory_geometry:
                self.remove_geometry(self.trajectory_geometry[name])
            
            colors = color
            if callable(color):
                # We get the color 
                color_intensities = color(self.observations)
                colors = Open3dUtils.get_RGB(np.array(color_intensities), minnorm=min(color_intensities), maxnorm=max(color_intensities))

            self.trajectory_geometry[name] = Open3dUtils.get_trajectory_geometry(*trajectory[:2], world_limits=self.worldgen.get_world_limits(), 
                                                                                 # If color is callable, we get the colors below
                                                                                 color=colors,
                                                                                 size=trajectory_size) 
            
            if isinstance(self.trajectory_geometry[name], list):
                for geom in self.trajectory_geometry[name]:
                    self.plot3d.add_geometry(geom)
            else:
                self.plot3d.add_geometry(self.trajectory_geometry[name])

            self.plot3d.update_step_geometries()

        replan()

        def register_parameter_change(key, setter):
            def update_params_and_plan_callable(mods):
                setter(mods)
                replan()
            
            self.plot3d.register_callback(key, lambda _, mods: update_params_and_plan_callable(mods)) # action is unused
        
        for tupl in key_replan:
            register_parameter_change(*tupl)
        

    def plot_distancefield(self,
                 num_arrows=9e3, dx=(0.0, 0.0), dy=(0.0, 0.0), dz=(0.0, 0.0), 
                 inflation=0.0):
        distancefield_gen = self.worldgen.get_distancefield(inflation=inflation)
        self.geovis = GeodesicVis(self.plot3d, distancefield_gen, num_arrows, dx, dy, dz)
        self.geovis.show_geodesic()

    
    def add_single_finished_planner(self, planner: PlannerRmp):
        # This function is mainly useful for debugging rollouts
        trajectory = planner.get_trajectory()

        geom = Open3dUtils.get_trajectory_geometry(*trajectory[:2], world_limits=self.worldgen.get_world_limits())
        self.plot3d.add_geometry(geom)
        self.plot3d.vis.poll_events()
        self.plot3d.update_renderer()

    def update_step_geometries(self):
        self.plot3d.update_step_geometries()

    def add_geometry(self, geometry):
        try:
            for geom in geometry:
                self.plot3d.add_geometry(geom)
        except:
            self.plot3d.add_geometry(geometry)

    def remove_geometry(self, geometry):
        try:
            for geom in geometry:
                self.plot3d.remove_geometry(geom)
        except:
            self.plot3d.remove_geometry(geometry)

    def add_geometry_with_hide_callback(self, geometry, key):
        geometry = copy.deepcopy(geometry)
        shown = [False]
        def toggle(action):
            if action == 0: return
            if shown[0]:
                self.remove_geometry(geometry)
            else:
                self.add_geometry(geometry)
            shown[0] = not shown[0]

        self.plot3d.register_callback(key, lambda action, _: toggle(action))


if __name__ == "__main__":
    # worldgen = CustomWorldgenFactory.HoleWall()
    # worldgen = CustomWorldgenFactory.BigWall()
    # worldgen = CustomWorldgenFactory.SingleCube()
    # worldgen = CustomWorldgenFactory.SnakeWalls()

    
    # worldgen = WorldgenRandomCpp(num_obstacles=500, seed=200)
    # worldgen = get_world([190, 401900011], WorldgenSettings.yaml_config_default(), 10.0)

    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(100, seed=0)
    worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=0)
    # worldgen = ProbabilisticWorldgenFactory.world25d(50, seed=0)

    planner3dvis = Planner3dVis(worldgen, distancefield=False)

    # Planner3dVisFactory.add_simple_target_planner(planner3dvis)
    Planner3dVisFactory.add_densely_sampled_data(planner3dvis, worldgen)

    # Planner3dVisFactory.add_expert_planner(planner3dvis, worldgen)

    planner3dvis.go()
    pass
