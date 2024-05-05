import numpy as np
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.planner.planner_params import PlannerParameters, RayObserverParameters, RaycastingCudaPolicyParameters, TargetPolicyParameters
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.camera_callback import PcaFollowingCameraCallback, PositionFollowingCameraCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import DelayAndHold, Mask
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.position_callback import TrajectoryCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.ray_callback import RayDecoderOutputCallback, RayTransition, UnprocessedRayCallback
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.camera import CameraFixed, FilterPositionCamera
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.movie_studio import Studio
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory

import rmp_dl.util.io as rmp_io

def transition_script(trajectory, md, start, length):
    # trajectory.register_geometry_getter(
    #     UnprocessedRayCallback()
    #         .Mask(start=0, stop=start)
    # )
    trajectory.register_geometry_getter(
        UnprocessedRayCallback(maxnorm=5.0)
            .Mask(start=start + length, stop=1000000)
            .DelayAndHold(delay_start=start, delay_length=length)
    )
    trajectory.register_geometry_getter(
        RayTransition(
            first=None,
            second=UnprocessedRayCallback(maxnorm=5.0),  # Set the correct ray length in the general_parameters.yml file
            start=start,
            length=length
        )
        .DelayAndHold(delay_start=start, delay_length=length)
    )

    trajectory.register_geometry_getter(
        TrajectoryCallback(radius=0.1)
            .DelayAndHold(delay_start=start, delay_length=length)
        )

    camera = CameraFixed(
        **{
            # You can get nice positions by plotting the world using any of the planner3dvis scripts 
            # (see 2023-11-28 - Presentation Video Worlds script for example)
            # setting the camera up that you like the view, and pressing ctr-c copies a json with the camera params.
			"pos" : [ 0.98914153920325198, -0.052233038081343768, -0.1373707580069882 ],
			"lookat" : [ 5.0, 6.6499999999999995, 7.5999999999999996 ],
			"up" : [ 0.060287120218931296, 0.99665715927441434, 0.055135922979150431 ],
			"zoom" : 0.8
        }
    )
    md.set_camera(camera)
    
    md.register_actor(trajectory)
    

def main():
    # This script needs max ray length to be set to 5.0 in the general_parameters.yml file
    worldgen = CustomWorldgenFactory.SimpleWorld()

    raycasting_cuda_params = RaycastingCudaPolicyParameters.from_yaml_general_config()
    planner_params = PlannerParameters.from_yaml_general_config()
    target_policy_params = TargetPolicyParameters.from_yaml_general_config()
    ray_observer_params = RayObserverParameters.from_yaml_general_config()
    ray_observer_params.maximum_ray_length = 5.0
    planner = PlannerFactory.baseline_shortdistance_ray_observer(target_policy_params, raycasting_cuda_params, ray_observer_params, planner_params, 
                                                                 save_rays=True)

    distancefield = worldgen.get_distancefield() if planner.requires_geodesic else None
    esdf = worldgen.get_esdf() if planner.requires_esdf else None
    planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf, distancefield)
    
    trajectory = PlannerTrajectoryActor(planner, name="planner")
    md = MovieDirector()

    transition_script(trajectory, md, start=250, length=200)

    world_mesh = Plot3D.get_world_geometry(worldgen)
    goal_geometry = Plot3D.get_sphere_geometry(worldgen.get_goal(), color=np.array([0, 1, 0]), radius=0.4)
    md.set_initial_geometries([world_mesh, goal_geometry])

    directory = rmp_io.resolve_directory("media/videos/presentation/02-ray_transition")
    directory = None

    Studio.do_run_and_make_video(md, output_directory=directory, filename="02-ray_transition.mp4", delete_images=False)

if __name__ == "__main__":
    main()