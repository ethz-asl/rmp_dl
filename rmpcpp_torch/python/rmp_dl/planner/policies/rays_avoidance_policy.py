import copy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from rmp_dl.planner.observers.observer import Observer
from rmp_dl.planner.state import State
from rmp_dl.util.halton_sequence import HaltonUtils
import torch
from policyBindings import PolicyValue, PolicyBase
from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, RayObserverParameters, RaycastingCudaPolicyParameters
from rmp_dl.planner.observers.ray_observer import RayObserver

import torch.nn.functional as F

from sklearn.decomposition import PCA

import torch.nn as nn

class RayAvoidance(nn.Module):
    # This is just used to convert the raycasts to a policy value. It is not learned, 
    # but torch is used for easy gpu operations, and also conversion to onnx format if necessary
    def __init__(self, params: RaycastingCudaPolicyParameters):
        super().__init__()
        self.training = False # This is a non-learned module, so we set this to false
        self.params = params

        self.n = params.N_sqrt ** 2

        # These are the unit vector endpoints of the halton sequence rays. Precompute them     
        points = HaltonUtils.get_ray_endpoints_from_halton_distances(np.ones(self.n))
        self.endpoints = torch.from_numpy(points)
        self.endpoints = self.endpoints.to(torch.float32)
        self.endpoints = self.endpoints.to(torch.device("cuda")) # (N, 3)


    def forward(self, rays, vel):
        # We convert the rays into a policy output, which is a 3D vector and a 3x3 matrix
        # This is according to the RMP avoidance policy as defined in the paper

        rays = rays.squeeze().unsqueeze(dim=-1) # make sure that we have (N, 1)

        d_hat = -self.endpoints

        f_rep  = self._alpha_rep(rays) * d_hat # (N, 3)
        f_damp = self._alpha_damp(rays) * self._g_obs(vel, d_hat) # (N, 3)

        f_obs = f_rep + f_damp # (N, 3)
        sf = self._softnorm(f_obs) # (N, 3)
        
        f_obs /= self.params.force_scale # Scale by number of rays
        w_r = self._w_r(rays).squeeze()  # (N)
        if self.params.metric:
            m = torch.einsum("ij, ik->ijk", sf, sf) # (N, 3, 3)
        else: 
            m = torch.eye(3).unsqueeze(0).repeat(self.n, 1, 1).to(torch.float32).to(device="cuda") # (N, 3, 3)
        
        A_obs = torch.einsum("i, ijk->ijk", w_r, m) # (N, 3, 3)
        A_obs /= self.params.metric_scale # Scale by number of rays

        metric_x_force_sum = torch.einsum("ijk, ik->j", A_obs, f_obs) # (3)
        metric_sum = torch.sum(A_obs, dim=0) # (3, 3)

        f = torch.linalg.pinv(metric_sum) @ metric_x_force_sum # (3, 3) @ (3) -> (3)
        A = metric_sum

        return f, A

    def _w_r(self, d):
        r = self.params.r
        w_r = 1 / r**2 * d**2 - 2 / r * d + 1
        w_r[d >= r] = 0
        return w_r

    def _g_obs(self, x_dot, d_hat): # (3), (N, 3) -> (N, 3)
        a = torch.max(torch.tensor(0), -torch.einsum('j, ij->i', x_dot, d_hat)) ** 2  
        b = d_hat 
        return torch.einsum("i, ij->ij", a, b)

    def _softnorm(self, x):
        z = torch.norm(x, dim=1, keepdim=True)
        c = self.params.c_softmax_obstacle
        return x / (z + c * torch.log(1 + torch.exp(- 2 * c * z)))

    def _alpha_rep(self, x):
        eta_rep = self.params.eta_rep
        v_rep = self.params.v_rep
        # Gets the scaling of the obstacle repulsion term
        return eta_rep * torch.exp(- x / v_rep)

    def _alpha_damp(self, x):
        # Gets the scaling of the obstacle damping term
        eta_damp = self.params.eta_damp
        v_damp = self.params.v_damp
        epsilon_damp = self.params.epsilon_damp
        return eta_damp / (x / v_damp + epsilon_damp)


class RaysAvoidancePolicy(PolicyBase):
    def __init__(self, 
                 params: RaycastingCudaPolicyParameters,
                 ray_observation_getter: Observer
                 ):
        PolicyBase.__init__(self) # Initialize c++ baseclass
        self.params = params
        self.ray_observation_getter = ray_observation_getter

        self.avoidance = RayAvoidance(params)

    # Overrides c++ base class method
    def evaluate_at(self, state: State) -> PolicyValue:
        rays = self.ray_observation_getter(state)

        f, metric = self.avoidance(rays, torch.from_numpy(state.vel.copy()).to(dtype=torch.float32, device="cuda"))
        
        return PolicyValue(f.cpu().numpy(), metric.cpu().numpy())


if __name__ == "__main__":
    avoidance = RayAvoidance(RaycastingCudaPolicyParameters.from_yaml_general_config())
    rays = torch.rand(1024, 1, device="cuda")
    vel = torch.rand(3, device="cuda")

    f, A = avoidance(rays, vel)