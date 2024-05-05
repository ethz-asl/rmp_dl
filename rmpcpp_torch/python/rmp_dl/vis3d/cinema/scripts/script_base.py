

import abc

from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.movie_director import MovieDirector


class Script:
    @abc.abstractmethod
    def __call__(self, md: MovieDirector, *args, **kwargs) -> None: ... 


class PlannerTrajectoryActorScript(Script):
    @abc.abstractmethod
    def __call__(self, md: MovieDirector, planner_trajectory_actor: PlannerTrajectoryActor) -> None: ... 


