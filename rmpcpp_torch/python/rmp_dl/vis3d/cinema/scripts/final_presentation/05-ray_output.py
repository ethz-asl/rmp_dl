from typing import Iterator
import numpy as np
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.camera_callback import PcaFollowingCameraCallback, PositionFollowingCameraCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import DelayAndHold, Mask
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.position_callback import TrajectoryCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.ray_callback import RayDecoderOutputCallback, RayTransition, UnprocessedRayCallback
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.actors.static_predictor.static_predictor import StaticPredictorActor, WorldDescription
from rmp_dl.vis3d.cinema.actors.world_builder.world_builder_actor import WorldBuilderActor
from rmp_dl.vis3d.cinema.actors.world_builder.world_sampler_actor import WorldSamplerActor
from rmp_dl.vis3d.cinema.camera import CameraFixed, FilterPositionCamera
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.movie_studio import Studio
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory

import rmp_dl.util.io as rmp_io
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

def get_circular_goal_translation_iterator(worldgen: WorldgenBase) -> Iterator[WorldDescription]:
    start = worldgen.get_start()
    goal = worldgen.get_goal()

    steps = 10000
    radius = 1.0

    def goal_translation_iterator() -> Iterator[WorldDescription]:
        yield WorldDescription(worldgen, goal, start)
        for i in range(steps):
            delta = np.array([np.cos(i/steps*2*np.pi * 100), np.sin(i/steps*2*np.pi * 100), 0]) * radius
            yield WorldDescription(None, goal + delta, start)
    return goal_translation_iterator()


if __name__ == "__main__":
    worldgen = CustomWorldgenFactory.SingleWall()

    model = ModelUtil.load_model("5msibfu3", "latest")

    world_description_iterator = get_circular_goal_translation_iterator(worldgen)

    actor = StaticPredictorActor(world_description_iterator, model=model, name="Static Actor")
    md = MovieDirector()
    md.register_actor(actor)
    
    camera = CameraFixed(
        **{
            # You can get nice positions by plotting the world using any of the planner3dvis scripts 
            # (see 2023-11-28 - Presentation Worlds script for example)
            # setting the camera up that you like the view, and pressing ctr-c copies a json with the camera params.
			"pos" : [ 0.95471338294830688, 0.14955286050992411, -0.25720866690821043 ],
			"lookat" : [ 3.1764170425895024, 4.4245781821622234, 2.3587974887169652 ],
			"up" : [ -0.1377443252762558, 0.98843417532687561, 0.063438016205336054 ],
			"zoom" : 0.69999999999999996
        }
    )
    md.set_camera(camera)

    directory = rmp_io.resolve_directory("media/videos/presentation/05-ray_output")
    directory = None

    Studio.do_run_and_make_video(md, output_directory=directory, filename="05-ray_output.mp4", delete_images=False)

