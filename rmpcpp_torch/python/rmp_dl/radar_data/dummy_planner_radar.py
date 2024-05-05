from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from rmp_dl.planner.planner import PlannerRmp
from rmp_dl.planner.planner_params import PlannerParameters
from rmp_dl.planner.state import PyState, State
from rmp_dl.radar_data.radar_data import RadarData
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen

from rmpPlannerBindings import PlannerRmpCpp

from nvbloxLayerBindings import EsdfLayer, TsdfLayer
from policyBindings import PolicyBase, PolicyValue


class DummyPlannerRadar:
    def __init__(self): 
        """
        Dummy Radar Planner class. Can use the same policies as the RMP planner, but does not actually plan. 
        (as the trajectory is predefined)
        The setup however is quite similar to the RMP planner.
        """
        self._setup_callable: Optional[Callable] = None  # Set by PlannerRadarBuilder 
        self._observations: PlannerRmp.DictWrapper = PlannerRmp.DictWrapper()
        self._is_setup = False
        self._step_idx = 0
        self._radar_data: RadarData = None # Set by setup_callable

        self._policies = []
    
    def get_trajectory_length(self):
        return len(self._radar_data)

    def add_policy(self, policy: PolicyBase):
        self._policies.append(policy)

    def reset(self):
        self._step_idx = 0

    def setup(self, data: RadarData):
        if self._setup_callable is None:
            raise ValueError("No setup callable provided, use the RadarPlannerBuilder class")
        
        self._is_setup = True
        self._setup_callable(self, data)

    def get_trajectory(self) -> np.ndarray:
        return self._radar_data.get_positions()
    
    def step(self, steps: int = 1) -> Tuple[Dict[str, Dict], bool]:
        """Run the planner for a number of steps. 

        Args:
            steps (int, optional): Number of steps to run. -1 means it will run to the end

        Returns:
            Tuple[Dict[str, Dict], bool]: Tuple of (observations, terminated). 
            Observations is a dictionary which maps observation id (for different policies) 
            to a dictionary of observations (observations within a single policy)
            In case of multiple steps, only the **last** observation is returned. 
        """
        if not self._is_setup:
            raise ValueError("Planner not setup, call setup() first")
        
        if self._step_idx >= len(self._radar_data) - 1:
            return None, True
        
        while steps != 0:
            obs, terminated = self._step()
            steps -= 1
            if terminated:
                break
        return obs, terminated

    
    def _step(self): 
        self._observations.dict = {} # Reset observations

        pos = self._get_pos()
        vel = self._get_vel()
        forward = self._get_forward_direction()

        vel = np.zeros_like(vel) # Velocity is kind of weird with the policy setup
        state: State = PyState(pos=pos, vel=vel, idx=self._step_idx, forward_direction=forward) 

        policy_values: List[PolicyValue] = [policy.evaluate_at(state) for policy in self._policies]
        f = PolicyValue.sum(policy_values).f_

        self._observations.dict.update({"state" : {"pos": pos, "vel": vel, "acc": f, "forward_direction": forward}})

        self._step_idx += 1

        return self._observations.dict, self._step_idx >= len(self._radar_data) - 1
    
    def _get_pos(self):
        return self._radar_data.get_position(self._step_idx)

    def _get_vel(self):
        return self._radar_data.get_velocity(self._step_idx)

    def _get_forward_direction(self):
        return self._radar_data.get_forward_direction(self._step_idx)
        