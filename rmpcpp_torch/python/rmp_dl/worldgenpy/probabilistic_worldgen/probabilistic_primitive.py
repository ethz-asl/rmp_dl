from __future__ import annotations

import abc
import re
from typing import Any, Callable, Dict, List, Tuple, Union, final
import numpy as np

from nvbloxSceneBindings import NvbloxSphere, NvbloxCube, NvbloxPrimitive


class ProbabilisticPrimitive(abc.ABC):
    def __init__(self, 
                 generator: np.random.Generator, 
                 world_limits: Tuple[np.ndarray, np.ndarray],
                 dependent_variables: Dict[str, Any] = {},
                 ):
        """This baseclass is used to define a probabilistic primitive.
        This is a primitive that has a distribution over its parameters.
        The parameters are registered using the register_params() method.
        The parameters are then resolved to their sampled value before using the _get() method.
        So the child class only has to register its numerical parameters and 
        implement the _get() method, and can use the parameters as if they are already sampled.
        """
        self.params: Dict[str, Any] = {}
        self.dependent_variables = dependent_variables
        self.generator = generator
        self.world_limits = world_limits

    def get_position_inside_world(self) -> np.ndarray:
        return self.generator.uniform(self.world_limits[0], self.world_limits[1])

    def register_params(self, **kwargs):
        """Registers parameters. This should be called in the constructor of the child class. 
        This automatically makes the parameters resolve to their sampled value according to the defined distribution (or value)
        when _get() is called"""
        for param_name, param_value in kwargs.items():
            self._register_param(param_name, param_value)

    def _register_param(self, param_name: str, param_value):
        if param_name in self.params:
            raise ValueError(f"Parameter {param_name} already registered")
        if hasattr(self, param_name):
            raise ValueError(f"Parameter {param_name} already exists as attribute")
        self.params[param_name] = param_value

    def _sample_and_set_params(self):
        # First we sample the dependent_variables
        sampled_dependent_variables: Dict[str, float] = {}
        for variable_name, variable in self.dependent_variables.items():
            sampled_dependent_variables[variable_name] = self._resolve_variable(variable, sampled_dependent_variables)

        # We then loop over the normal registered parameters
        sampled_variables: Dict[str, float] = {}
        for param_name, param_value in self.params.items():
            sampled_variables[param_name] = self._resolve_variable(param_value, sampled_dependent_variables)

        # We then set the parameters
        for param_name, param_value in sampled_variables.items():
            self._set_param(param_name, param_value)

    def _set_param(self, param_name, param_value):
        # We set the parameter as attribute. Some danger for weird bugs here, but I think there's enough checking
        # in the rest of the code that it is reasonably safe. I like this more than keeping e.g. some dict
        # as it makes the code in the child classes a lot cleaner
        setattr(self, param_name, param_value)

    def _resolve_variable(self, variable: Union[str, float, int], sampled_dependent_variables: Dict[str, Union[float, int]]) -> Union[float, int]:
        """Resolves a variable. If it is already a float or int, it just returns it. 
        If the variable is a string, it will:

        - If $variable == DEPENDENT($name), it will look up $name in the sampled_dependent_variables and return its value. 
          Raises an exception if it doesn't exist
        - If $variable == CALL(*args), it will parse the call and call the function with the arguments.

        """
        if isinstance(variable, float) or isinstance(variable, int):
            return variable
        
        function_name, args = self._parse_call(variable)

        if function_name == "DEPENDENT":
            if len(args) != 1:
                raise ValueError(f"DEPENDENT takes exactly one argument, got {args}")
            if args[0] not in sampled_dependent_variables:
                raise ValueError(f"DEPENDENT variable {args[0]} not found in sampled_dependent_variables")
            return sampled_dependent_variables[args[0]]

        return self._resolve_sampling_call(function_name, args)
        

    def _parse_call(self, call_string: str):
        # We match on the function name and arguments (group1, group2)
        match = re.match(r'(\w+)\((.*?)\)$', call_string)

        if not match:
            raise ValueError(f"Failed to parse {call_string}")

        function_name = match.group(1)
        arguments = [arg.strip() for arg in match.group(2).split(",")] # We split on , and strip the whitespace
        def try_to_float(f):
            try: 
                return float(f)
            except: 
                return f
        arguments = [try_to_float(arg) for arg in arguments] # We convert to float if possible

        return function_name, arguments
    
    def _resolve_sampling_call(self, function_name, function_arguments) -> float:
        if function_name == "UNIFORM":
            if len(function_arguments) != 2:
                raise ValueError(f"UNIFORM takes exactly two arguments, got {function_arguments}")
            return self._sample_uniform(*function_arguments)
        elif function_name == "NORMAL":
            if len(function_arguments) != 2:
                raise ValueError(f"NORMAL takes exactly two arguments, got {function_arguments}")
            return self._sample_normal(*function_arguments)
        else:
            raise ValueError(f"Unknown sampling type{function_name}")

    def _sample_uniform(self, min_value, max_value) -> float:
        return self.generator.uniform(min_value, max_value)

    def _sample_normal(self, mean, std) -> float:
        return self.generator.normal(mean, std)

    @final
    def get(self) -> NvbloxPrimitive:
        self._sample_and_set_params()
        return self._get()

    @abc.abstractmethod
    def _get(self) -> NvbloxPrimitive: ...

    @staticmethod
    def resolve_obstacle(obstacle_type) -> type[ProbabilisticPrimitive]:
        if obstacle_type == "Sphere":
            return Sphere
        elif obstacle_type == "Box":
            return Box
        else:
            raise ValueError(f"Unknown obstacle type {obstacle_type}")

class Sphere(ProbabilisticPrimitive):
    def __init__(self, 
                 radius: Union[str, float], 
                 **kwargs):
        super().__init__(**kwargs)
        super().register_params(
            radius=radius
        )
        # I just put this here so that the static analysis doesn't complain because we are using setattr in the baseclass
        self.radius: float

    def _get(self) -> NvbloxPrimitive:
        pos = self.get_position_inside_world()
        return NvbloxSphere(pos, self.radius)

class Box(ProbabilisticPrimitive):
    def __init__(self, 
                 length_x: Union[str, float], 
                 length_y: Union[str, float], 
                 length_z: Union[str, float], 
                 **kwargs):
        super().__init__(**kwargs)
        super().register_params(
            length_x = length_x, 
            length_y = length_y, 
            length_z = length_z, 
        )
        # I just put this here so that the static analysis doesn't complain because we are using setattr in the baseclass
        self.length_x: float
        self.length_y: float
        self.length_z: float

    def _get(self) -> NvbloxPrimitive:
        pos = self.get_position_inside_world()
        return NvbloxCube(pos, np.array([self.length_x, self.length_y, self.length_z]))