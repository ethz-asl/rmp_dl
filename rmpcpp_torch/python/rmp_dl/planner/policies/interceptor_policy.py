from typing import Callable, Optional
import numpy as np
from rmp_dl.planner.state import State
from policyBindings import PolicyValue, PolicyBase
from nvbloxLayerBindings import TsdfLayer

class PolicyInterceptor(PolicyBase):
    """This class is used to intercept the output of the raycasting
    obstacle avoidance policy to python. The output can then be used
    for e.g. training or visualization. 
    """
    def __init__(self, 
                 intercepted_policy: PolicyBase, 
                 observation_callback: Callable[[str, dict], None], 
                 name: Optional[str]=None,
                 active=True):
        PolicyBase.__init__(self)
        
        self.intercepted = intercepted_policy
        self.name = name
        self.last_metric: np.ndarray = np.identity(3)
        self.last_force: np.ndarray = np.zeros(3)
        self.observation_callback = observation_callback
        self.active = active

    def evaluate_at(self, state: State) -> PolicyValue:
        value: PolicyValue = self.intercepted.evaluate_at(state)
        self.last_metric = value.A_
        self.last_force = value.f_
        if self.name is not None:
            self.observation_callback(self.name, {"A": self.last_metric.copy(), "f": self.last_force.copy()})

        if self.active:
            return value

        return PolicyValue(np.zeros(3), np.zeros((3, 3)))
    
