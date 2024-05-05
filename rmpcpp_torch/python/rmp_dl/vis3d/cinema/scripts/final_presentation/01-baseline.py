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
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory

import rmp_dl.util.io as rmp_io

def transition_script(trajectory, md):
    # trajectory.register_geometry_getter(
    #     UnprocessedRayCallback()
    #         .Mask(start=0, stop=start)
    # )
    trajectory.register_geometry_getter(
        TrajectoryCallback(radius=0.03)
        )

    camera = CameraFixed(
        **{
			"pos" : [ -0.97113125970275782, -0.0097975648849338362, -0.23834446532374148 ],
			"lookat" : [ 4.9613467970617435, 5.2671343463443625, 5.1604840432120662 ],
			"up" : [ 0.045533922443432504, 0.973171269213432, -0.22553124547262501 ],
			"zoom" : 0.23999999999999957
        }
    )
    md.set_camera(camera)
    
    md.register_actor(trajectory)
    

def main():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(150, seed=1212, with_bounds=True)

    planner = PlannerFactory.baseline_labeled()

    distancefield = worldgen.get_distancefield() if planner.requires_geodesic else None
    esdf = worldgen.get_esdf() if planner.requires_esdf else None
    planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf, distancefield)
    
    trajectory = PlannerTrajectoryActor(planner, name="planner")
    md = MovieDirector()

    transition_script(trajectory, md)

    world_mesh = Plot3D.get_world_geometry(worldgen)
    goal_geometry = Plot3D.get_sphere_geometry(worldgen.get_goal(), color=np.array([0, 1, 0]), radius=0.2)
    # start_geometry = Plot3D.get_sphere_geometry(worldgen.get_start(), color=np.array([0, 0, 1]), radius=0.2)
    md.set_initial_geometries([world_mesh, goal_geometry])

    directory = rmp_io.resolve_directory("media/videos/presentation/01-baseline")
    directory = None

    Studio.do_run_and_make_video(md, output_directory=directory, filename="01-baseline.mp4", delete_images=False)

if __name__ == "__main__":
    main()