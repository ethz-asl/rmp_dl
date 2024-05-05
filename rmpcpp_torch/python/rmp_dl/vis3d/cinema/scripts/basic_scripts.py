import numpy as np
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.camera_callback import PcaFollowingCameraCallback, PositionFollowingCameraCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.position_callback import TrajectoryCallback
from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.ray_callback import RayDecoderOutputCallback, RayTransition, UnprocessedRayCallback
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.camera import CameraFixed, FilterPositionCamera
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.scripts.script_base import PlannerTrajectoryActorScript


class OutputDecoderVisScript(PlannerTrajectoryActorScript):
    def __call__(self, md: MovieDirector, planner_trajectory_actor: PlannerTrajectoryActor) -> None:
        planner_trajectory_actor.register_geometry_getter(
            RayDecoderOutputCallback()
        )

        planner_trajectory_actor.register_geometry_getter(
            TrajectoryCallback()
            )

        camera = FilterPositionCamera(filter_cutoff=0.005)
        md.set_camera(camera)
        
        planner_trajectory_actor.register_general_observation_callback(
            PcaFollowingCameraCallback(camera)
        )

        md.register_actor(planner_trajectory_actor)
        
class OutputDecoderVisScriptFollowCam(PlannerTrajectoryActorScript):
    def __call__(self, md: MovieDirector, planner_trajectory_actor: PlannerTrajectoryActor) -> None:
        planner_trajectory_actor.register_geometry_getter(
            RayDecoderOutputCallback()
        )

        planner_trajectory_actor.register_geometry_getter(
            TrajectoryCallback()
            )

        camera = FilterPositionCamera(filter_cutoff=0.01)
        md.set_camera(camera)
        
        planner_trajectory_actor.register_general_observation_callback(
            PositionFollowingCameraCallback(camera)
        )

        md.register_actor(planner_trajectory_actor)
        

class RayVisScriptFollowCam(PlannerTrajectoryActorScript):
    def __call__(self, md: MovieDirector, planner_trajectory_actor: PlannerTrajectoryActor) -> None:
        planner_trajectory_actor.register_geometry_getter(
            UnprocessedRayCallback()
        )

        planner_trajectory_actor.register_geometry_getter(
            TrajectoryCallback()
            )

        camera = FilterPositionCamera(filter_cutoff=0.015)
        md.set_camera(camera)
        
        planner_trajectory_actor.register_general_observation_callback(
            PositionFollowingCameraCallback(camera)
        )

        md.register_actor(planner_trajectory_actor)



class RayVisScriptStaticCam(PlannerTrajectoryActorScript):
    def __call__(self, md: MovieDirector, planner_trajectory_actor: PlannerTrajectoryActor) -> None:
        planner_trajectory_actor.register_geometry_getter(
            UnprocessedRayCallback()
        )

        planner_trajectory_actor.register_geometry_getter(
            TrajectoryCallback()
            )

        camera = CameraFixed(
            pos=np.array([10.0, 10.0, 10.0]), 
            lookat=np.array([5.0, 5.0, 5.0]), 
            up=np.array([0.0, 1.0, 0.0])
        )

        md.set_camera(camera)
        md.register_actor(planner_trajectory_actor)


class TrajectoryVisScriptStaticCam(PlannerTrajectoryActorScript):
    def __call__(self, md: MovieDirector, planner_trajectory_actor: PlannerTrajectoryActor) -> None:
        planner_trajectory_actor.register_geometry_getter(
            TrajectoryCallback(radius=0.1)
            )

        camera = CameraFixed(
            pos=np.array([15.0, 9.0, 7.0]), 
            lookat=np.array([5.0, 4.0, 5.0]), 
            up=np.array([0.0, 1.0, 0.0]), 
            zoom=0.92
        )

        md.set_camera(camera)
        md.register_actor(planner_trajectory_actor)

