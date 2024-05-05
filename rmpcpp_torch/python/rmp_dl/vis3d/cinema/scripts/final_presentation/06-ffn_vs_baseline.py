import numpy as np
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.camera_callback import PcaFollowingCameraCallback, PositionFollowingCameraCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import DelayAndHold, Mask
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.position_callback import TrajectoryCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.ray_callback import RayDecoderOutputCallback, RayTransition, UnprocessedRayCallback
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.camera import CameraFixed, FilterPositionCamera
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.movie_studio import Studio
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory

import rmp_dl.util.io as rmp_io

def trajectory_script(trajectory: PlannerTrajectoryActor, md):
    trajectory.register_geometry_getter(
        RayDecoderOutputCallback()
    )
    trajectory.register_geometry_getter(
        TrajectoryCallback(radius=0.01)
    )

    camera = FilterPositionCamera(height=0.4, filter_cutoff=0.005)
    md.set_camera(camera)
    
    camera_transition_frame = 500
    trajectory.register_general_observation_callback(
        PositionFollowingCameraCallback(camera)
            .Mask(start=0, stop=camera_transition_frame)
    ) 
    trajectory.register_general_observation_callback(
        PcaFollowingCameraCallback(camera)
            .Mask(start=camera_transition_frame, stop=1000000)
    )
    
    md.register_actor(trajectory)
    

def main():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=119000048, with_bounds=True)

    model = ModelUtil.load_model("5msibfu3", "latest")
    planner = PlannerFactory.learned_planner_with_ray_observer_and_inactive_expert(model)

    distancefield = worldgen.get_distancefield() if planner.requires_geodesic else None
    esdf = worldgen.get_esdf() if planner.requires_esdf else None
    planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf, distancefield)
    
    trajectory = PlannerTrajectoryActor(planner, name="planner")
    md = MovieDirector()

    trajectory_script(trajectory, md)

    world_mesh = Plot3D.get_world_geometry(worldgen)
    goal_geometry = Plot3D.get_sphere_geometry(worldgen.get_goal(), color=np.array([0, 1, 0]), radius=0.2)
    # start_geometry = Plot3D.get_sphere_geometry(worldgen.get_start(), color=np.array([0, 0, 1]), radius=0.2)


    # We also visualize the baseline in black
    baseline_planner = PlannerFactory.baseline_labeled()
    baseline_planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf, distancefield)

    baseline_planner.step(-1)
    baseline_trajectory_geometry = Open3dUtils.get_trajectory_geometry(*baseline_planner.get_trajectory()[:2], world_limits=worldgen.get_world_limits(), color=np.array([0, 0, 0]))
    
    md.set_initial_geometries([world_mesh, goal_geometry, baseline_trajectory_geometry])

    directory = rmp_io.resolve_directory("media/videos/presentation/06-ffn_vs_baseline2")
    # directory = None

    Studio.do_run_and_make_video(md, output_directory=directory, filename="06-ffn_vs_baseline2.mp4", delete_images=True)

if __name__ == "__main__":
    main()