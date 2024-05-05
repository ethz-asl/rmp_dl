

from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.camera_callback import PcaFollowingCameraCallback, PositionFollowingCameraCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.position_callback import TrajectoryCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.ray_callback import RayDecoderOutputCallback, RayTransition, UnprocessedRayCallback
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.camera import FilterPositionCamera
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.scripts.script_base import PlannerTrajectoryActorScript


class TransitionScript(PlannerTrajectoryActorScript):
    def __init__(self, start: int, length: int) -> None:
        self.start = start
        self.length = length
        
    def __call__(self, md: MovieDirector, planner_trajectory_actor: PlannerTrajectoryActor) -> None:
        start = self.start
        length = self.length
        
        planner_trajectory_actor.register_geometry_getter(
            UnprocessedRayCallback()
                .Mask(start=0, stop=start)
        )
        planner_trajectory_actor.register_geometry_getter(
            RayDecoderOutputCallback()
                .Mask(start=start, stop=1000000)
                .DelayAndHold(delay_start=start, delay_length=length)
        )
        planner_trajectory_actor.register_geometry_getter(
            RayTransition(
                first=UnprocessedRayCallback(),
                second=RayDecoderOutputCallback(),
                start=start,
                length=length
            )
            .DelayAndHold(delay_start=start, delay_length=length)
        )

        planner_trajectory_actor.register_geometry_getter(
            TrajectoryCallback()
                .DelayAndHold(delay_start=start, delay_length=length)
            )

        camera = FilterPositionCamera(filter_cutoff=0.012)
        md.set_camera(camera)
        
        planner_trajectory_actor.register_general_observation_callback(
            PositionFollowingCameraCallback(camera)
                .Mask(start=0, stop=start)
                .DelayAndHold(delay_start=start, delay_length=length)
        ) 
        planner_trajectory_actor.register_general_observation_callback(
            PcaFollowingCameraCallback(camera)
                .Mask(start=start, stop=1000000)
                .DelayAndHold(delay_start=start, delay_length=length)
        )

        md.register_actor(planner_trajectory_actor)
        