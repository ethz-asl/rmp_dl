import numpy as np
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.planner.planner_params import PlannerParameters, RayObserverParameters, RaycastingCudaPolicyParameters, TargetPolicyParameters
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.camera_callback import PcaFollowingCameraCallback, PositionFollowingCameraCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import DelayAndHold, Mask
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.position_callback import TrajectoryCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.ray_callback import RayDecoderOutputCallback, RayTransition, UnprocessedRayCallback
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.camera import CameraFixed, CameraRotating, FilterPositionCamera
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.movie_studio import Studio
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory

import rmp_dl.util.io as rmp_io

def rotation_script(md):
    camera = CameraRotating(
        center= np.array([5.0, 5.0, 5.0]),
        radius= 40.0,
        speed=0.02,
        up= np.array([0.0, 1.0, 0.0])
    )
    md.set_camera(camera)
    md.register_actor(camera.get_actor())

def main():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=1293)
    # worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=8343)

    md = MovieDirector()

    planner = PlannerFactory.learned_planner_from_wandb_id("g2j8uxxd", "max_sum50_decoder", "last")
    distancefield = worldgen.get_distancefield() if planner.requires_geodesic else None
    esdf = worldgen.get_esdf() if planner.requires_esdf else None
    planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf, distancefield)   

    planner.step(-1)
    trajectory_geometry = Open3dUtils.get_trajectory_geometry(
        *planner.get_trajectory()[:2], world_limits=worldgen.get_world_limits(), color=[1., 0., 0.], 
        size=0.1)   

    rotation_script(md)

    world_mesh = Plot3D.get_world_geometry(worldgen)
    initial_geometries = [world_mesh]
    initial_geometries.extend(trajectory_geometry)
    md.set_initial_geometries(initial_geometries)

    directory = rmp_io.resolve_directory("media/videos/presentationRSS/rotating_example_sb")
    # directory = None

    Studio.do_run_and_make_video(md, output_directory=directory, filename="sb.mp4", delete_images=True)

if __name__ == "__main__":
    main()