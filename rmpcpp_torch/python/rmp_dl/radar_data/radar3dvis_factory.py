
import numpy as np
from rmp_dl.radar_data.planner_radar_factory import PlannerRadarFactory

import open3d as o3d
from rmp_dl.radar_data.radar3dvis import Radar3DVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.vis3d.vis3d import Plot3D

class Radar3DVisFactory:
    @staticmethod
    def add_radar(radar_3dvis: Radar3DVis, model, name, color=[1, 0, 0]):
        radar_dummy_planner_create = PlannerRadarFactory.radar_planner

        params = {
            "model": model,
        }
        radar_3dvis.add_planner(name, radar_dummy_planner_create, params, color=color)


        radar_3dvis.register_step_geometry_change(Radar3DVisFactory.add_radar_points_visualization_callback(
            name, radar_points_name="radar_step_points"
        ))

        radar_3dvis.register_step_geometry_change(Planner3dVisFactory.add_pos_visualization_callback(name))

        radar_3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(
            name, "radar_converter", radar_3dvis.increment_and_get_step_geometry_idx()))
        
        radar_3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(
            name, "ray_interpolator", radar_3dvis.increment_and_get_step_geometry_idx()))
        
        radar_3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_pred_visualization_callback(
                name, mod_idx=radar_3dvis.increment_and_get_step_geometry_idx()))

        # Visualize forward direction only
        radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
            name, radar_3dvis.increment_and_get_global_geometry_idx(), 
            lambda d: d["state"]["forward_direction"], color=[0, 0, 0], arrow_scale=0.15)
        )

        # Visualize forward direction together with model direction prediction
        idx = radar_3dvis.increment_and_get_global_geometry_idx()
        radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
            name, idx, 
            lambda d: d["state"]["forward_direction"], color=[0, 0, 0], arrow_scale=0.15)
        )
        radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
            name, idx, 
            lambda d: d["learned_policy"]["geodesic_prediction"], color=[1, 0, 0], arrow_scale=0.15)
        )
        
        # # Visualize forward direction together with model policy output
        # idx = radar_3dvis.increment_and_get_global_geometry_idx()
        # radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
        #     name, idx, 
        #     lambda d: d["state"]["forward_direction"], color=[0, 0, 0], arrow_scale=0.15)
        # )
        # radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
        #     name, idx, 
        #     lambda d: d["learned_policy_interceptor"]["f"], color=[1, 1, 0], arrow_scale=0.15)
        # )

        # Visualize velocity together with avoidance prediction
        idx = radar_3dvis.increment_and_get_global_geometry_idx()
        radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
            name, idx, 
            lambda d: d["state"]["vel"], color=[0, 0, 1], arrow_scale=0.15, constant_scale=False)
        )
        radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
            name, idx, 
            lambda d: d["avoidance_policies"]["f"], color=[0, 1, 0], arrow_scale=0.15, constant_scale=False)
        )

        # Visualize forward_direction_together with full prediction
        idx = radar_3dvis.increment_and_get_global_geometry_idx()
        radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
            name, idx, 
            lambda d: d["state"]["forward_direction"], color=[0, 0, 0], arrow_scale=0.15)
        )
        radar_3dvis.register_global_geometry_change(Planner3dVisFactory.add_full_trajectory_arrow_visualization_callback(
            name, idx, 
            lambda d: d["state"]["acc"], color=[0, 1, 1], arrow_scale=0.15, constant_scale=False)
        )



    @staticmethod
    def add_radar_points_visualization_callback(name, radar_points_name):
        def callback(observations: dict, idx: int, step_geometry_modifier, idx2: int):
            if idx2 <= 0:
                return
            points = []
            for i in range(max(0, idx - idx2 + 1), idx + 1):
                if i >= len(observations[name]):
                    break
                radar_points = observations[name][i][radar_points_name]["radar_points"]
                points.append(radar_points)
            if len(points) == 0:
                return []
            radar_points = np.vstack(points)

            geometries = []
            for point in radar_points:
                geometries.append(Plot3D.get_sphere_geometry(point, radius=0.1, color=[0, 0, 0]))
            

            return geometries
        
        return callback