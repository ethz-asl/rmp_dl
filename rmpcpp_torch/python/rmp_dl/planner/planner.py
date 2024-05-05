from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from rmp_dl.planner.planner_params import PlannerParameters
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen

from rmpPlannerBindings import PlannerRmpCpp

from nvbloxLayerBindings import EsdfLayer, TsdfLayer
from policyBindings import PolicyBase

class PlannerRmp:
    class DictWrapper:
        # We wrap the dict class such that we can weak reference it: https://docs.python.org/3/library/weakref.html
        # Also we wrap it and not subclass the dict class,
        # such that code outside this module can still hold a reference to the wrapper dict object once reset.
        def __init__(self):
            self.dict = {}

    def __init__(self, params: PlannerParameters): 
        """Rmp planner class. Use the static methods in PlannerFactory to generate new planner instances. (Or use the PlannerBuilder)
        The static methods need to define the `_planning_callable` member variable. 
        This callable will be called in case of a planning run. 
        This class has been set up this way to separate construction of this class
        (e.g. setting up the policies) with the actual planning runs, 
        because the policies can depend on parameters only known at planning time 
        (e.g. goal location, tsdf).
        One note:
        We reconstruct the PlannerRmpCpp object every time, instead of making it possible to 
        remove the policies on the cpp side. This is because we have no way of deleting policies instantiated
        by python (so python policies derived from policy base), as we use 
        py::keep_alive<> to bind the policy to the PlannerRmpCpp class. So the only way to 
        make sure that the python policies are destroyed is to re-instantiate the planner class completely.

        The observations are set by callbacks which are added to policies. 

        Args:
            params (PlannerParameters, optional): _description_. Defaults to PlannerParameters().
        """
        self._params: PlannerParameters = params
        self._planner = PlannerRmpCpp(params.to_cpp())
        self._planner_setup_callable: Optional[Callable] = None  # Set by PlannerBuilder in planner_factory.py
        self._observations: PlannerRmp.DictWrapper = PlannerRmp.DictWrapper()
        self.requires_geodesic = False
        self.requires_esdf = False
        self._is_setup = False
    
    def get_trajectory_length(self):
        return len(self.get_trajectory()[0])

    def get_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Get the trajectory (tuple of positions, velocities and accelerations). 

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (pos, vec, acc), each Nx3 where N is the number of traj. points
        """
        return self._planner.get_trajectory()
    
    def success(self):
        return self._planner.success()
    
    def collided(self):
        return self._planner.collided()
    
    def add_policy(self, policy: PolicyBase):
        self._planner.add_policy(policy)

    def get_pos(self) -> np.ndarray:
        return self._planner.get_pos()

    def get_vel(self) -> np.ndarray:
        return self._planner.get_vel()

    def get_acc(self) -> np.ndarray:
        return self._planner.get_acc()

    def get_previous_pos(self) -> np.ndarray:
        return self._planner.get_previous_pos()
    
    def get_previous_vel(self) -> np.ndarray:
        return self._planner.get_previous_vel()
    
    def get_previous_acc(self) -> np.ndarray:
        return self._planner.get_previous_acc()

    def diverged(self) -> bool:
        return self._planner.diverged()

    def setup(self, start: np.ndarray, target: np.ndarray, 
             tsdf: TsdfLayer, esdf: TsdfLayer=None,
             geodesic: Optional[DistanceFieldGen]=None) -> None:
        """Setup a planning run. Planner requires both tsdf and esdf
        but uses only 1 of them (generally), based on the initialized policies and planners. 

        Args:
            start (np.ndarray): Starting location
            target (np.ndarray): Target location
            tsdf (TsdfLayer): Nvblox Tsdf layer
            esdf (EsdfLayer): Nvblox Tsdf layer. The esdf layer uses the TSDF type, but should have a very large truncation distance. 
                This used to use the actual ESDF layer as well, but they are less accurate as they measuer distances in terms of voxels. 
            geodesic (GeodesicLayer, optional): Geodesic field. Defaults to None.
        """
        if self._planner_setup_callable is None:
            raise RuntimeError("Planner not initialized correctly. \
                               Make sure to use the planner builder in planner_factory.py")

        self._is_setup = True
        
        # See the planner builder in planner_factory.py
        self._planner_setup_callable(self, start, target, tsdf, esdf, geodesic)
    
    def step(self, steps: int = 1, terminate_if_stuck: bool = False) -> Tuple[Dict[str, Dict], bool]:
        """Run the planner for a number of steps. 

        Args:
            steps (int, optional): Number of steps to run. -1 means it will run until succes, collision, or when it exceeds total max steps of trajectory. 
            Defaults to 1.

        Returns:
            Tuple[Dict[str, Dict], bool]: Tuple of (observations, terminated). 
            Observations is a dictionary which maps observation id (for different policies) 
            to a dictionary of observations (observations within a single policy)
            In case of multiple steps, only the **last** observation is returned. 
        """
        if not self._is_setup:
            raise RuntimeError("Planner not setup. Make sure to call setup() first")
        
        self._observations.dict = {} # Reset observations
        self._planner.step(steps)

        # The policies during a step work on the previous state, so we save the previous state as observations
        # (the state from before the step above)
        # This is not done before the step above, because if we are doing multiple steps we only save the last observations, 
        # so we want to retrieve this previous state
        pos, vel, acc = self.get_previous_pos(), self.get_previous_vel(), self.get_previous_acc()
        self._observations.dict.update({"state" : {"pos": pos, "vel": vel, "acc": acc}})

        terminated = self._planner.success() or self._planner.collided() or self._planner.diverged() 
        terminated |= terminate_if_stuck and self._determine_stuck()

        return self._observations.dict, terminated
    
    def _determine_stuck(self):
        # This is very hacky, but not really used anyway. 
        vel, acc = self.get_vel(), self.get_acc()
        if np.linalg.norm(vel) < 0.05 and np.linalg.norm(acc) < 0.05:
            # print("STUCK")
            return True
        
        return False


        