
import abc
from typing import Callable, Dict, List, Optional
from attr import dataclass
import numpy as np
from rmp_dl.planner.planner import PlannerRmp
from rmp_dl.vis3d.cinema.actor import ActorBase

from rmp_dl.vis3d.cinema.actors.planner.observation_callbacks.observation_callback import GeometryGetter, ObservationCallback, UpdateGeometries


class PlannerTrajectoryActor(ActorBase):
    """Actor that rolls out a trajectory
    We register callbacks that consume the observation and return geometries to add and remove
    We keep returning geometries from the callbacks (even if we have finished rolling out the trajectory)
    until all callbacks are finished. 

    Args:
        ActorBase (_type_): _description_
    """
    def __init__(self, planner: PlannerRmp, **kwargs):
        super().__init__(**kwargs)
        self.planner = planner
        self.planner_terminated = False

        self.geometries_to_add = []
        self.geometries_to_remove = []

        self.geometry_getters: List[ObservationCallback[UpdateGeometries]] = []
        self.active_geometry_getters: List[ObservationCallback[UpdateGeometries]] = []        

        self.general_observation_callbacks: List[ObservationCallback[None]] = []
        self.active_general_observation_callbacks: List[ObservationCallback[None]] = []

    def has_next_step(self) -> bool:
        return len(self.active_geometry_getters) > 0 
    
    def next_step(self):
        if not self.has_next_step():
            return
        
        observation = None
        if not self.planner_terminated:
            observation, self.planner_terminated = self.planner.step(terminate_if_stuck=False)
        # We can still have delayed geometries or callbacks (e.g. the camera doing things at the end), 
        # so we continue calling those with None in case the planner has terminated
        self._get_geometries(observation)
        self._do_general_callbacks(observation)
    
    def _do_general_callbacks(self, observation):
        deactivate_callbacks = []
        for i, callback in enumerate(self.active_general_observation_callbacks):
            try:
                callback(observation)
            except IndexError:
                deactivate_callbacks.append(i)
                continue
        
        # Remove callbacks that raised an IndexError
        for i in deactivate_callbacks:
            self.active_general_observation_callbacks.pop(i)

    def _get_geometries(self, observation):
        self.geometries_to_add = []
        self.geometries_to_remove = []
        deactivate_callbacks = []
        for i, callback in enumerate(self.active_geometry_getters):
            try:
                update_geometries = callback(observation)
            except IndexError:
                deactivate_callbacks.append(i)
                continue
            if update_geometries is None: # Could be that it is deactivated
                continue
            self.geometries_to_add.extend(update_geometries.to_add)
            self.geometries_to_remove.extend(update_geometries.to_remove)

        # Remove callbacks that raised an IndexError
        for i in reversed(sorted(deactivate_callbacks)):
            self.active_geometry_getters.pop(i)

    def get_geometries_to_add(self):
        return self.geometries_to_add
    
    def get_geometries_to_remove(self):
        return self.geometries_to_remove

    def register_geometry_getter(self, getter: ObservationCallback[UpdateGeometries]):
        self.geometry_getters.append(getter)
        self.active_geometry_getters.append(getter)
        return self # So we can chain calls
    
    def register_general_observation_callback(self, callback: ObservationCallback[None]):
        self.general_observation_callbacks.append(callback)
        self.active_general_observation_callbacks.append(callback)
        return self # So we can chain calls
