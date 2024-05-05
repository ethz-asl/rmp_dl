
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from policyBindings import State as CppState

""" 
We define a pystate here. The original rmp code uses a c++ evaluate_at function that uses a State object as parameter. 
This however is not really flexible, as that object can only contain pos and vel. 
With the radar code, we also want to include the idx of the radar step. So here we define 
a type alies sort of thing that says that it can be either type. 
A lot of this should be refactored in general I think, probably it is a good idea to remove the c++ integrator
and move that to python, and defining a more flexible input for the policies
"""

@dataclass(frozen=False)
class PyState:
    pos: np.ndarray
    vel: np.ndarray
    idx: Optional[int] = None # Can also have state without the idx, so we set it to none by default.
    forward_direction: Optional[np.ndarray] = None

State = Union[CppState, PyState]