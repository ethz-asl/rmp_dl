from operator import itemgetter
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase


class Sweep(PipelineObjectBase):
    def __init__(self, 
                 sweep_type: str, 
                 sweep_type_params: dict, 
                 sweep_variable_name: str, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sweep_type = sweep_type
        self.sweep_type_params = sweep_type_params
        self.sweep_variable_name = sweep_variable_name

        self.call_count = 0
        self.resolve_sweep_type(sweep_type, sweep_type_params) # Check if sweep type is valid to raise error early
        
        # Reset the call count
        self.call_count = 0
    
    def on_epoch_start(self):
        super().on_epoch_start()
        self.call_count += 1

    def __iter__(self) -> Iterable[List[Tuple[str, Any]]]:
        # If there is no input node we are an entry node, and we just return our own sweep
        if len(self._get_inputs()) == 0:
            yield from self._iterate()
            return
        
        # If we have an input, we generate a nested sweep
        for input_sweep in self._get_input():
            for sweep in self._iterate():
                yield input_sweep + sweep  # Combine the two lists of tuples

    def _iterate(self) -> Iterable[List[Tuple[str, Any]]]:
        # Iterates over the sweep that this object represents
        # So it returns a list of tuples, where each tuple is a variable name and its value
        # It is a list of tuples as it can be combined with other sweeps
        for value in self.resolve_sweep_type(self.sweep_type, self.sweep_type_params):
            yield [(self.sweep_variable_name, value)]
                
    
    def resolve_sweep_type(self, sweep_type, sweep_type_params) -> Iterable[Any]:
        if sweep_type == "steps":
            start, stop, step = itemgetter("start", "stop", "step")(sweep_type_params)
            return np.arange(start, stop, step)
        elif sweep_type == "values":
            return iter(sweep_type_params)
        elif sweep_type == "jumps":
            start, jump_size, num_jumps = itemgetter("start", "jump_size", "num_jumps")(sweep_type_params)
            diff = (self.call_count - 1) * jump_size * num_jumps
            return range(start + diff, start + jump_size * num_jumps + diff, jump_size)
        else: 
            raise ValueError(f"Unknown sweep type: {sweep_type}")
        
