from __future__ import annotations

import abc
import copy
import dataclasses
from typing import Any, Dict, Generic, List, Optional, TypeVar
from attr import dataclass

import open3d as o3d

# Wanted to do some experimenting with generics in python. Could probably do this 
# in a better way though. 
# Point was to have these callbacks with different return types that have these generic operations on them such as masking
# or delays. As these callbacks are used to fetch geometries and camera views, 
# Could also have made it that the callback is in charge of setting whatever it wants to set (so another callback basically instead of returning something)
# But this would be kind of a cyclic way of defining things. Anyway idk, kind of experimenting here. 


ReturnType = TypeVar('ReturnType')  # Define a TypeVar for the return type
class ObservationCallback(Generic[ReturnType], abc.ABC):
    @abc.abstractmethod
    def __call__(self, observation: Optional[Dict]) -> Optional[ReturnType]: ...
    
    # You can define operations on the callbacks, that alter the observation dictionary while it is getting passed through. 
    # Note however that by chaining, the order of the operations is reversed.
    def chain(self, observation_callback_operation_type: type[ObservationCallbackOperation[ReturnType]], *args, **kwargs) -> ObservationCallback[ReturnType]:
        return observation_callback_operation_type(self, *args, **kwargs)

    def Mask(self, *args, **kwargs) -> ObservationCallback[ReturnType]:
        return self.chain(Mask, *args, **kwargs)
    
    def DelayAndHold(self, *args, **kwargs) -> ObservationCallback[ReturnType]:
        return self.chain(DelayAndHold, *args, **kwargs)

@dataclass
class UpdateGeometries:
    to_add: List[o3d.geometry.Geometry]
    to_remove: List[o3d.geometry.Geometry]

    def __iter__(self):
        return iter((self.to_add, self.to_remove))
    
class GeometryGetter(ObservationCallback[UpdateGeometries]):
    @abc.abstractmethod
    def __call__(self, observation: Optional[Dict]) -> UpdateGeometries: ...

class GeneralCallback(ObservationCallback[None]):
    @abc.abstractmethod
    def __call__(self, observation: Optional[Dict]) -> None: ...

class ObservationCallbackOperation(ObservationCallback[ReturnType]):
    @abc.abstractmethod
    def __init__(self, callback: ObservationCallback[ReturnType], *args, **kwargs): ...

class Mask(ObservationCallbackOperation[ReturnType]):
    def __init__(self, callback: ObservationCallback[ReturnType], start: int, stop: int):
        self.start = start
        self.stop = stop
        self.callback = callback

        self.count = -1

    def __call__(self, observation: Dict) -> Optional[ReturnType]:
        self.count += 1
        if self.count < self.start:
            # The caller knows that None just means to do nothing in case it expects something (e.g. in case ReturnType is UpdateGeometries)
            return None

        if self.count >= self.stop:
            raise IndexError("Mask has been exceeded")

        return self.callback(observation)
    

class DelayAndHold(ObservationCallbackOperation[ReturnType]):
    def __init__(self, callback: ObservationCallback[ReturnType], delay_start: int, delay_length: int):
        self.callback = callback

        self.delay_start = delay_start
        self.delay_length = delay_length

        self.delayed_observations: List[Optional[Dict]] = []
        self.hold_observation: Optional[Dict] = None
        self.count = -1

    def __call__(self, observation: Optional[Dict]) -> Optional[ReturnType]:
        self.count += 1
        if self.count < self.delay_start:
            return self.callback(observation)

        if self.count == self.delay_start:
            self.hold_observation = copy.deepcopy(observation)
            return self.callback(observation)

        self.delayed_observations.append(copy.deepcopy(observation))

        if self.count < self.delay_start + self.delay_length:
            return self.callback(self.hold_observation)

        return self.callback(self.delayed_observations.pop(0))
