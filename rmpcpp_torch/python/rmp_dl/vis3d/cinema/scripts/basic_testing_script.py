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
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory

import rmp_dl.util.io as rmp_io

def transition_script(trajectory, md, start, length):
    trajectory.register_geometry_getter(
        UnprocessedRayCallback()
            .Mask(start=0, stop=start)
    )
    trajectory.register_geometry_getter(
        RayDecoderOutputCallback()
            .Mask(start=start, stop=1000000)
            .DelayAndHold(delay_start=start, delay_length=length)
    )
    trajectory.register_geometry_getter(
        RayTransition(
            first=UnprocessedRayCallback(),
            second=RayDecoderOutputCallback(),
            start=start,
            length=length
        )
        .DelayAndHold(delay_start=start, delay_length=length)
    )

    trajectory.register_geometry_getter(
        TrajectoryCallback()
            .DelayAndHold(delay_start=start, delay_length=length)
        )

    camera = FilterPositionCamera(filter_cutoff=0.015)
    md.set_camera(camera)
    
    trajectory.register_general_observation_callback(
        PositionFollowingCameraCallback(camera)
            .Mask(start=0, stop=start)
            .DelayAndHold(delay_start=start, delay_length=length)
    ) 
    trajectory.register_general_observation_callback(
        PcaFollowingCameraCallback(camera)
            .Mask(start=start, stop=1000000)
            .DelayAndHold(delay_start=start, delay_length=length)
    )

    md.register_actor(trajectory)
    

def main():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=171254)

    model = ModelUtil.load_model("3byflajt", "latest")
    model.set_output_decoder_from_factory("max_sum512_decoder")
    planner = PlannerFactory.learned_planner_with_ray_observer_and_inactive_expert(model)

    distancefield = worldgen.get_distancefield() if planner.requires_geodesic else None
    esdf = worldgen.get_esdf() if planner.requires_esdf else None
    planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf, distancefield)
    
    trajectory = PlannerTrajectoryActor(planner, name="planner")
    md = MovieDirector()

    transition_script(trajectory, md, start=350, length=50)

    world_mesh = Plot3D.get_world_geometry(worldgen)
    md.set_initial_geometries([world_mesh])

    directory = rmp_io.resolve_directory("media/videos/transition_script")

    md.go()

if __name__ == "__main__":
    main()