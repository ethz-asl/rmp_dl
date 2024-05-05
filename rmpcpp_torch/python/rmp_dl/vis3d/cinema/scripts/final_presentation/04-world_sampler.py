import numpy as np
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.camera_callback import PcaFollowingCameraCallback, PositionFollowingCameraCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import DelayAndHold, Mask
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.position_callback import TrajectoryCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.ray_callback import RayDecoderOutputCallback, RayTransition, UnprocessedRayCallback
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.actors.world_builder.world_builder_actor import WorldBuilderActor
from rmp_dl.vis3d.cinema.actors.world_builder.world_sampler_actor import WorldSamplerActor
from rmp_dl.vis3d.cinema.camera import CameraFixed, FilterPositionCamera
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.movie_studio import Studio
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory

import rmp_dl.util.io as rmp_io

def main():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=1215, goal=np.array([7.,6.,8.]))

    world_builder = WorldSamplerActor(worldgen, num_arrows=1024, scale=0.4)
    md = MovieDirector()

    md.register_actor(world_builder)

    world_mesh = Plot3D.get_world_geometry(worldgen)
    goal_geometry = Plot3D.get_sphere_geometry(worldgen.get_goal(), color=np.array([0, 1, 0]), radius=0.4)
    md.set_initial_geometries([world_mesh, goal_geometry])

    camera = CameraFixed(
        **{
            # You can get nice positions by plotting the world using any of the planner3dvis scripts 
            # (see 2023-11-28 - Presentation Worlds script for example)
            # setting the camera up that you like the view, and pressing ctr-c copies a json with the camera params.
			"pos" : [ 0.50138681175424871, 0.053359269223190196, 0.86357631590200346 ],
			"lookat" : [ 5.0350885746525815, 4.9076269202049403, 4.0975439852582731 ],
			"up" : [ 0.27108454960361505, 0.93815478272408681, -0.21535730918203569 ],
			"zoom" : 0.94000000000000017
        }
    )
    md.set_camera(camera)
    
    directory = rmp_io.resolve_directory("media/videos/presentation/04-world_sampler")
    # directory = None

    Studio.do_run_and_make_video(md, output_directory=directory, filename="04-world_sampler.mp4", delete_images=False)

if __name__ == "__main__":
    main()