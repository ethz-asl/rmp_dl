
import os
import time
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from rmp_dl.radar_data.dummy_planner_radar import DummyPlannerRadar
from rmp_dl.radar_data.radar_data import RadarData
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.vis3d.vis3d import Plot3D
import open3d as o3d
from rmp_dl.vis3d.vis3d_stepped import Plot3DStepped


class Radar3DVis:
    def __init__(self, radar_data: RadarData):
        self.plot3d = Plot3DStepped()
        self.params = {}
        self._radar_data = radar_data
        self._plot_radar()
        self.trajectory_geometry = {}
    
    def _plot_radar(self):
        positions = self._radar_data.get_positions()
        o3d_points = o3d.utility.Vector3dVector(positions)
        pc = o3d.geometry.PointCloud(o3d_points)
        pc.paint_uniform_color([0, 0, 0])
        self.plot3d.add_geometry(pc)
        
        radar_measurement = self._radar_data.get_pointcloud()
        o3d_radar_points = o3d.utility.Vector3dVector(radar_measurement)
        radar_pc = o3d.geometry.PointCloud(o3d_radar_points)

        color_intensities = self._radar_data.get_doppler()
        colors = Open3dUtils.get_RGB(np.array(color_intensities), minnorm=min(color_intensities), maxnorm=max(color_intensities))

        radar_pc.colors = o3d.utility.Vector3dVector(colors)

        self.plot3d.create_origin_boxes(0.2)

        self.plot3d.add_geometry(radar_pc)

    @property
    def observations(self):
        return self.plot3d.observations
    
    def increment_and_get_step_geometry_idx(self):
        return self.plot3d.increment_and_get_step_geometry_idx()

    def register_step_geometry_change(self, getter):
        self.plot3d.register_step_geometry_change(getter)

    def register_global_geometry_change(self, getter):
        self.plot3d.register_global_geometry_change(getter)

    def increment_and_get_global_geometry_idx(self):
        return self.plot3d.increment_and_get_global_geometry_idx()

    def go(self):
        self.plot3d.run()

    def add_planner(self, name, planner_create: Callable[..., DummyPlannerRadar], 
                    planner_params: dict,
                    key_replan: List[Tuple[str, Callable[[None], float], Callable[[float], None], float]] = [], 
                    color=Optional[Union[List, Callable]], 
                    trajectory_size=None) -> None:
        """Add a planner to the visualizer. 
        The planner create callable is used to create the planner, passing the planner_params dict as kwargs to the constructor.
        The key_replan list can be used to create parameter change callbacks which trigger a replan, mapped to keys during 3d visualization. 
        Every tuple is of (key, getter, setter, increment). The key is the key to press to trigger the callback. The getter method gets the 
        current parameter value, while the setter method is used to set the new value. The increment is the amount to increment/decrement the parameter.


        Args:
            name (str): Name of the planner. Not very important, just make sure that every call to this function has a unique name.
            planner_create (Callable): Constructor to create the planner. Arguments to the constructor can be supplied with kwargs
            key_replan (List[Tuple[str, str, str, float]]): List of tuples to update increment/decrement the planner parameters and replan. 
                Each entrty is a tuple of (key, getter, setter, increment). 
            color (Optional[Union[List, Callable]], optional): Color of the trajectory. Can be a list of [r, g, b] values, or a callable which takes the observations dict and returns a list of colors. 
                Defaults to None. 
            trajectory_size: Size of the trajectory. If None (recommended) a pointcloud geometry is used to visualize. If you want to manually set the size, 
                sphere geometries are used which are less performant, so only use this for making figures for a report or something. 
        """
        self.plot3d.current_step_geometry_modifier_steps = 0
        if name in self.params:
            raise RuntimeError("Name is already in use")
        self.params[name] = planner_params
        self.observations[name] = []

        def replan():
            planner: DummyPlannerRadar = planner_create(**self.params[name])
            planner.setup(self._radar_data)
            start = time.time()
            while True:
                observations, terminated = planner.step(1)
                self.observations[name].append(observations)
                if terminated:
                    break
            t = time.time() - start
            print(f"Planner {name} took {t} seconds to plan, integration speed: {planner.get_trajectory_length() / t} hz")
            trajectory = planner.get_trajectory()

            if name in self.trajectory_geometry:
                self.plot3d.vis.remove_geometry(self.trajectory_geometry[name])
            
            self.trajectory_geometry[name] = Open3dUtils.get_trajectory_geometry(trajectory, 
                                                                                 # If color is callable, we get the colors below
                                                                                 color=None if callable(color) else color,
                                                                                 size=trajectory_size) 
            if callable(color):
                color_intensities = color(self.observations)
                colors = Open3dUtils.get_RGB(np.array(color_intensities), minnorm=min(color_intensities), maxnorm=max(color_intensities))
                # If we are using size in get_trajectory_geometry above, we are plotting sphere geometries instead of a point cloud, 
                # and we need to set every colour of every sphere manually
                if isinstance(self.trajectory_geometry[name], list):
                    for i, geom in enumerate(self.trajectory_geometry[name]):
                        geom.paint_uniform_color(colors[i])
                else:
                    self.trajectory_geometry[name].colors = o3d.utility.Vector3dVector(colors)
            if isinstance(self.trajectory_geometry[name], list):
                for geom in self.trajectory_geometry[name]:
                    self.plot3d.add_geometry(geom)
            else:
                self.plot3d.add_geometry(self.trajectory_geometry[name])

            self.plot3d.update_step_geometries()

        replan()

        def register_parameter_change(key, getter, setter, increment):
            resolve_increment = lambda mods: increment if mods == 0 else -increment # Increment if no modifiers (e.g. shift), decrement otherwise
            def update_params_and_plan_callable(mods):
                setter(getter() + resolve_increment(mods))
                replan()
            
            self.plot3d.register_callback(ord(key), lambda _, mods: update_params_and_plan_callable(mods)) # action is unused
        
        for tupl in key_replan:
            register_parameter_change(*tupl)

if __name__ == "__main__":
    path = os.path.join('experiment7_urban', '02_urban_night_H_processed.bag')
    radar_data = RadarData(path)

    radar_vis = Radar3DVis(radar_data)
    radar_vis.go()
