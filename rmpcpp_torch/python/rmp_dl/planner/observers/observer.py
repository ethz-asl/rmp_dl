import abc
import copy
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Union, final
from attr import dataclass

import numpy as np
from rmp_dl.planner.state import State


class Observer:
    """An observer is a class that can be used to provide common observations to policies. 
    The observer provides a __call__ method that accepts a state and returns any type. 
    The observer can then be passed to policies, such that the policies can __call__ the observer to get the observation.
    
    An example is to get ray observations. The observer would be a class that accepts a state, and returns the ray observation.
    Because multiple policies can call the observer, the result is cached. We have to do it like this because call order of policies
    is not really known as it is done in the cpp code. So we can't really e.g. register a observer as a passive policy
    because we don't know whether it will be called before or after the other policies. We could enforce call order, but prefer not to.

    Because an observer generally also uses data that is only known at planning time such as the TSDF, use the builder to 
    register the observer. The builder takes care of passing through params like the TSDF to the observer, 
    and the other policies can use the observer. 
    """
    def __init__(self):
        self._last_call_state: State = None
        self._last_call_result = None

    @final
    def __call__(self, state: State) -> Any:
        if self._last_call_state is None or \
                not np.equal(self._last_call_state.pos, state.pos).all() or \
                not np.equal(self._last_call_state.vel, state.vel).all():
            self._last_call_state = state
            self._last_call_result = self._get_observation(state)

        # TODO: I don't think a deepcopy is actually necessary here. 
        # As I all observers (should be/) are pure-functions. Especially the CUDA tensor stuff. 
        return self._last_call_result

    @abc.abstractmethod
    def _get_observation(self, state: State) -> Any: ...

