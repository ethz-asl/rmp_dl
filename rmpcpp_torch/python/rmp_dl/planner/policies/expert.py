
from typing import Callable, Optional, Tuple
import numpy as np
from rmp_dl.planner.planner_params import TargetPolicyParameters
from rmp_dl.planner.policies.common import SimpleTarget
from rmp_dl.planner.state import State
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

from policyBindings import PolicyValue, PolicyBase
from nvbloxLayerBindings import EsdfLayer

class ExpertPolicy(PolicyBase):
    def __init__(self, 
                 target: np.ndarray,
                 geodesic: DistanceFieldGen,
                 target_policy_rmp_params: TargetPolicyParameters,
                 observation_callback: Optional[Callable[[str, dict], None]] = None):
        PolicyBase.__init__(self) # Initialize c++ baseclass
        self.observation_callback = observation_callback
        self.target = target
        self.simple_target = SimpleTarget(target_policy_rmp_params.alpha, target_policy_rmp_params.beta, target_policy_rmp_params.c_softmax)
        self.geodesic = geodesic 

    # Overrides c++ base class method
    def evaluate_at(self, state: State) -> PolicyValue:
        geodesic = self.geodesic.get_gradient_interpolate(state.pos).flatten()
        if np.linalg.norm(geodesic) < 1e-6:
            geodesic = geodesic / 1e-6
        else:
            geodesic = geodesic / np.linalg.norm(geodesic)

        if self.observation_callback is not None:
            self.observation_callback("expert_policy", {"geodesic": geodesic.copy().flatten()}) 

        # Mutiply with distance, theres a softnorm inside simple_target, so this makes sure it goes to 0 close to the goal. 
        dist = np.linalg.norm(self.target - state.pos)
        geodesic *= dist

        if dist < 0.4: # The geodesic field is unreliable close to the goal, so we just go straight to the goal
            geodesic = self.target - state.pos

        f = self.simple_target(geodesic, state.vel)

        return PolicyValue(f, (np.identity(3)))