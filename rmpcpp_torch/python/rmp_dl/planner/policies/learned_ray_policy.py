import copy
import os, sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from rmp_dl.learning.lightning_module import RayLightningModule
from rmp_dl.learning.model import RayModel, RayModelDirectionConversionWrapper
from rmp_dl.learning.model_parts.model_with_skip import RecurrentModelWithSkipAdd
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.policies.common import SimpleTarget
from rmp_dl.planner.state import State
import torch
from policyBindings import PolicyValue, PolicyBase
from nvbloxLayerBindings import TsdfLayer
from kernelBindings import get_rays
from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, RayObserverParameters
from rmp_dl.planner.observers.ray_observer import RayObserver

import torch.nn.functional as F

from sklearn.decomposition import PCA

class LearnedRayPolicy(PolicyBase):
    def __init__(self, 
                 model: RayModelDirectionConversionWrapper,
                 learned_policy_rmp_params: LearnedPolicyRmpParameters,
                 ray_observation_getter: Observer,
                 target: np.ndarray,
                 observation_callback: Optional[Callable[[str, dict], None]] = None,
                 track_lstm_output_difference=False, 
                 ):
        PolicyBase.__init__(self) # Initialize c++ baseclass
        self.callback = observation_callback
        self.target = target
        self.simple_target = SimpleTarget(learned_policy_rmp_params.alpha, learned_policy_rmp_params.beta, learned_policy_rmp_params.c_softmax)
        self.model = model
        try:
            self.model.eval()
        except: pass
        self.track_lstm_output_difference = track_lstm_output_difference

        # If we use a ray decoder, we track the ray weights
        self._temp_ray_weights = None
        
        # We track some extra things if the observation logging callback is enabled. 
        # If not, we don't do anything to avoid overhead
        if observation_callback is not None:
            self._setup_ray_prediction_intercept_hook(model)
            self._setup_recurrent_hook()
    
        self.hidden_state = None
        self.ray_observation_getter = ray_observation_getter

    # Overrides c++ base class method
    @torch.no_grad()
    def evaluate_at(self, state: State) -> PolicyValue:
        rays = self.ray_observation_getter(state)
        
        rel_pos = self._resolve_rel_pos(state)
        vel = state.vel.copy()
        dist = np.linalg.norm(rel_pos)
        vel = torch.from_numpy(vel).float().to(device="cuda:0")
        rel_pos = torch.from_numpy(rel_pos).float().to(device="cuda:0")

        geodesic = self._get_prediction(rays, rel_pos, vel)

        if self.callback is not None:
            self._do_callbacks(rel_pos, vel, geodesic, rays)

        # Mutiply with distance, theres a softnorm inside simple_target, so this makes sure it goes to 0 close to the goal. 
        geodesic *= dist

        if dist < 0.4: # No point in using the model when this close to the goal
            geodesic = self.target - state.pos

        f = self.simple_target(geodesic, state.vel)

        metric = np.identity(3)

        return PolicyValue(f, metric)

    def _get_prediction(self, rays, rel_pos, vel):
        geodesic, self.hidden_state = self.model(
                rays.unsqueeze(0).unsqueeze(0), rel_pos.unsqueeze(0).unsqueeze(0), vel.unsqueeze(0).unsqueeze(0), hiddens=self.hidden_state
            )
        geodesic = geodesic.squeeze().cpu().detach().numpy()
        return geodesic

    def _do_callbacks(self, rel_pos, vel, geodesic, rays):
        # Geodesic prediction by the network
        d = {"geodesic_prediction": geodesic.copy()}

        # Ray outputs before decoding into cartesian geodesic prediction
        if self._temp_ray_weights is not None:
            output_ray_pred = self._temp_ray_weights.cpu().detach().squeeze().numpy()
            d.update({"output_ray_predictions": output_ray_pred.copy()})
        
        # Track how much the recurrent network changes the norm of the latent space
        if self._temp_lstm_diff_norm is not None:
            d.update({"recurrent_diff_norm": copy.deepcopy(self._temp_lstm_diff_norm)})

        # Track how much difference there is between a prediction with and without lstm hidden state set. 
        if self.track_lstm_output_difference and self._temp_ray_weights is not None:
            # None for hidden state means lstm with zero initialized hidden state
            no_lstm = self.model.forward(
                rays.unsqueeze(0).unsqueeze(0), rel_pos.unsqueeze(0).unsqueeze(0), vel.unsqueeze(0).unsqueeze(0), hiddens=None
            )[0].squeeze().cpu().detach().numpy()
            d.update({"no_hidden_state_output_ray_predictions": no_lstm.copy()})

        d.update({"goal": self.target}) # During plotting we want this info
        self.callback("learned_policy", d) 

    
    def _setup_ray_prediction_intercept_hook(self, model):
        # Yeah lots of nested things, should be cleaned up
        if isinstance(model.model, RayModel):
            model.model.register_forward_hook(self._output_ray_prediction_hook)
        else:
            model.model.model.register_forward_hook(self._output_ray_prediction_hook)

    def _setup_recurrent_hook(self):
        # We track how much of a difference the lstm has made in additive skip add architecture
        self._temp_lstm_diff_norm = {}

        def hook(input, output, name):
            input = input[0].cpu().detach().numpy()
            output = output[0].cpu().detach().numpy() # [1] is hidden state, [0] is the output
            self._temp_lstm_diff_norm[name] = np.linalg.norm(output - input) / np.linalg.norm(input)

        for name, module in self.model.named_modules():
            if isinstance(module, RecurrentModelWithSkipAdd):
                module.register_forward_hook(lambda _, input, output: hook(input, output, name))


    def _output_ray_prediction_hook(self, module: RayModel, input, output):
        if module.ray_decoding_learned:
            self._temp_ray_weights = output[0] # [1] is the hidden state, [0] is the ray weights

    def _resolve_rel_pos(self, state: State):
        if self.target is not None:
            return self.target - state.pos.copy()
        
        # If target is none, we use the forward direction as the goal
        try: 
            forward = state.forward_direction
        except: 
            # If forward direction is not defined, we are probably using a cpp state and planner. 
            # Throw exception
            raise ValueError("No target defined, and no forward direction defined in state. Cannot continue")
        
        if np.linalg.norm(forward) < 1e-8:
            rel_pos = np.array([0, 0, 1])
        else:
            rel_pos = forward / np.linalg.norm(forward) * 12.0  # Large number, so it's far away, this gets normalized inside the model

        return rel_pos