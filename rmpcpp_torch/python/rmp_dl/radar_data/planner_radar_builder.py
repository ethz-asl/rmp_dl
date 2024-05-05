from __future__ import annotations
import abc
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, final
import weakref
import numpy as np
from rmp_dl.planner.planner import PlannerRmp
from rmp_dl.planner.planner_builder import DummyPolicy

from rmp_dl.planner.planner_params import PlannerParameters
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.policies.interceptor_policy import PolicyInterceptor
from rmp_dl.radar_data.dummy_planner_radar import DummyPlannerRadar
from rmp_dl.radar_data.radar_data import RadarData

from rmpPlannerBindings import PlannerRmpCpp
from policyBindings import PolicyBase


class PlannerRadarBuilder:
    def __init__(self):
        self.policy_tuples = []
        self.observer_tuples = []

    def register_observer(self, observer_constructor: Callable[[Any], Observer],
                  observer_name: Optional[str] = None,
                  additional_params: Optional[List[Union[str, Tuple[str, str]]]] = None):
        """Add an observer to the planner. A observer is a class that can be used to provide common observations to policies.
        The observer provides a __call__ method that accepts a state and returns any type.
        A policy (or another observer) can get the observer passed to its constructor by adding the observer_name to its additional_params set.
        By providing a tuple ($observer_name, $param_name) to the additional_params set, the observer with $observer_name will be passed to the policy
        as a parameter with name $param_name. If no tuple is supplied, an identity mapping is assumed (e.g. for tsdf, target etc. this is usually the case)

        Args:
            observer_constructor (Callable[[Any], PolicyBase]): Constructor for the observer. 
            observer_name (str): Name of the observer 
            additional_params (Optional[set], optional): Additional parameters such as TSDF that should be passed to the constructor. Defaults to None.
        """
        if additional_params is None:
            additional_params = set()
        self.observer_tuples.append((observer_constructor, observer_name, additional_params))

    def add_policy(self, policy_constructor: Callable[[Any], PolicyBase],  
                   additional_params: Optional[List[Union[str, Tuple[str, str]]]] = None, 
                   intercept=False,
                   interceptor_name=None) -> PlannerRadarBuilder:
        """Add a policy to the planner.

        Args:
            policy_constructor (callable): Callable that creates a policy, that accepts kwargs for the additional params set in $additional_params
            additional_params (set, optional): Which of the possible additional params should be passed (as kwargs) to the policy upon planner 
                construction. Possibilities are: esdf, tsdf, geodesic, target and observation_callback, or any of the parameters set by the obsevers. 
                If a parameter should be passed as a different name, pass a tuple with ($param_name, $target_name). Defaults to None.
            intercept (bool, optional): Whether to intercept the policy with a PolicyInterceptor. Defaults to False.
            interceptor_name (str, optional): The name the interceptor uses to store the output in the observations dict, 
                required if intercept is True. Defaults to None.
            active (bool, optional): Whether the policy is active. intercepted. Defaults to True.
        """
        if additional_params is None:
            additional_params = []

        self.policy_tuples.append((policy_constructor, additional_params, intercept, interceptor_name))
        return self


    def build(self) -> PlannerRmp:
        planner = DummyPlannerRadar()
        
        def get_additional_params_mapping(additional_params: List[Union[str, Tuple[str, str]]], ) -> Dict[str, str]:
            d = {}
            for param in additional_params:
                if isinstance(param, str):
                    d[param] = param
                else:
                    d[param[0]] = param[1]
            return d

        # Because some parameters for the policies are only known at planning time (in this case only radar data, 
        # for the rmp planner it was more) we need to create a callable that sets up the planner with the correct parameters.
        def planning_setup_callable(planner: DummyPlannerRadar, radar_data: RadarData):
            planner._radar_data = radar_data
            # See comments in PlannerBuilder (the rmp builder) for why weakref was used there. Not 100% necessary to also keep it here
            # but why not
            weakref_observations = weakref.ref(planner._observations)
            observation_callback  = lambda id, output: weakref_observations().dict.update({id: output})

            additional_params_dict = {
                'radar_data': radar_data,
                'observation_callback': observation_callback,
            }

            # We add the observers to the additional params dict. 
            # Note that because we add it immediately to the dict, we could have 'nested' observers;
            # an observer that uses another observer, as long as the order is correct.
            observers = []
            for observer in self.observer_tuples:
                observer_constructor, observer_name, additional_params = observer
                additional_params_mapping = get_additional_params_mapping(additional_params)
                filtered_dict = {additional_params_mapping[k]: v 
                                 for k, v in additional_params_dict.items() if k in additional_params_mapping}
                observer = observer_constructor(**filtered_dict)
                if observer_name is not None:
                    if observer_name in additional_params_dict:
                        raise ValueError(f"Observer with name {observer_name} already exists")
                    additional_params_dict[observer_name] = observer
                observers.append(observer)

            # Dummy policy makes sure that all the observers are always called, even if an observer is not used by any policy
            # (e.g. in the case of just saving observations)
            planner.add_policy(DummyPolicy(observers=observers))

            # see the add_policy() method for the order of these params. 
            for policy_constructor, additional_params, \
                intercept, interceptor_name in self.policy_tuples:
                
                # We map the additional params to the correct names, and check if all required params are provided.
                additional_params_mapping = get_additional_params_mapping(additional_params)
                filtered_dict = {additional_params_mapping[k]: v 
                                 for k, v in additional_params_dict.items() if k in additional_params_mapping}
                if len(filtered_dict) != len(additional_params):
                    raise ValueError(f"Not all required additional params were provided for policy {str(policy_constructor)}")
                policy = policy_constructor(**filtered_dict)

                # Check if the policy needs to be intercepted (intercepted policies automatically put their output in the observations dict)
                # Intercepted policies can also be set to inactive with the active flag set to False. 
                if intercept is True:
                    policy = PolicyInterceptor(intercepted_policy=policy, observation_callback=observation_callback, 
                                               name=interceptor_name)
                
                planner.add_policy(policy)


        planner._setup_callable = planning_setup_callable
        return planner