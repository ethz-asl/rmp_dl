from __future__ import annotations
import copy
import os
from typing import Any, Callable, Dict, List, Optional, Union
import weakref
import numpy as np
from rmp_dl.learning.model import RayModel, RayModelDirectionConversionWrapper, RayModelStaticSizeWrapper
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.learning.util import WandbUtil
from rmp_dl.planner.observers.partial_ray_observer import PartialRayObserver
from rmp_dl.planner.observers.ray_accumulator import RayAccumulator
from rmp_dl.planner.observers.ray_noisifier import RayNoiser
from rmp_dl.planner.observers.ray_observation_combiner import RayObservationCombiner
from rmp_dl.planner.observers.ray_observation_interpolator import RayObservationInterpolator
from rmp_dl.planner.observers.ray_saver import RaySaver
from rmp_dl.planner.observers.ray_stochastic_downsampler import RayStochasticDownsampler
from rmp_dl.planner.planner import PlannerRmp
from rmp_dl.planner.planner_builder import PlannerBuilder

from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, PlannerParameters, RayObserverParameters, RaycastingCudaPolicyParameters, TargetPolicyParameters
from rmp_dl.planner.policies.expert import ExpertPolicy
from rmp_dl.planner.policies.learned_ray_policy import LearnedRayPolicy
from rmp_dl.planner.observers.ray_observer import RayObserver
from rmp_dl.planner.policies.rays_avoidance_policy import RaysAvoidancePolicy
import torch


from policyBindings import PolicyBase, RaycastingCudaPolicy, SimpleEsdfPolicy, SimpleTargetPolicy

import rmp_dl.util.io as rmp_io

class PlannerFactory: 
    @staticmethod
    def baseline_planner_with_default_params():
        raycasting_cuda_params = RaycastingCudaPolicyParameters.from_yaml_general_config()
        planner_params = PlannerParameters.from_yaml_general_config()
        target_policy_params = TargetPolicyParameters.from_yaml_general_config()
        return PlannerFactory.baseline(target_policy_params, raycasting_cuda_params, planner_params)

    @staticmethod
    def learned_planner_from_wandb_id(wandb_id: str, decoder_method: str, alias="last", 
                                      *args, **kwargs) -> PlannerRmp:
        with rmp_io.TempDirectories(f"/models/{wandb_id}-{alias}-{os.getpid()}") as download_dir:
            file = WandbUtil.download_model(wandb_id, alias, download_dir)

            raycasting_cuda_params = RaycastingCudaPolicyParameters.from_yaml_general_config(expert=False)
            
            learned_policy_rmp_params = LearnedPolicyRmpParameters.from_yaml_general_config()
            learned_policy_ray_observer_params = RayObserverParameters.from_yaml_general_config()
            
            planner_params = PlannerParameters.from_yaml_general_config()
            
            return PlannerFactory.learned_planner_from_checkpoint_path(
                file, wandb_id, decoder_method, 
                raycasting_cuda_policy_params=raycasting_cuda_params,
                learned_policy_rmp_params=learned_policy_rmp_params,
                ray_observer_params=learned_policy_ray_observer_params,
                planner_params=planner_params,
                *args, **kwargs
            )

    @staticmethod
    def learned_planner_from_checkpoint_path(
        path: str, wandb_id: str, decoder_method: str, 
        size: Optional[float] = None,
        *args, **kwargs, 
        ) -> PlannerRmp:
        model = ModelUtil.get_model_from_checkpoint_file(path, wandb_id)
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(decoder_method)
        if size is not None:
            model = RayModelStaticSizeWrapper(model, size)
        
        return PlannerFactory.learned_planner_minimal(model, *args, **kwargs)
        
    @staticmethod
    def learned_planner_with_ray_observer_from_checkpoint_path(
        path, 
        wandb_id, 
        decoder_method: str, 
        *args, **kwargs,
        ):
        model = ModelUtil.get_model_from_checkpoint_file(path, wandb_id)
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(decoder_method)

        return PlannerFactory.learned_planner_with_ray_observer(model, *args, **kwargs)

    @staticmethod
    def learned_planner_minimal(
        model: RayModelDirectionConversionWrapper,
        raycasting_cuda_policy_params: RaycastingCudaPolicyParameters, 
        learned_policy_rmp_params: LearnedPolicyRmpParameters, 
        ray_observer_params: RayObserverParameters,
        planner_params: PlannerParameters,
        cpp_policy=True,
        add_ray_noise=False,
        ray_noise_params=None,
        ):
        """Generate a planner with learned target policy. This planner does not save intermediate observations
        """
        learned_policy_params = {
            "model": model,
            "learned_policy_rmp_params": learned_policy_rmp_params, 
        }

        builder = PlannerBuilder(planner_params)
        
        ray_observer_params_policy = {
            "ray_observer_params": ray_observer_params
        }
        
        builder.register_observer(lambda **planning_time_params: RayObserver(**ray_observer_params_policy, **planning_time_params), 
                                  observer_name="ray_observer",
                                  additional_params=['tsdf'])
        
        ray_observer_name = 'ray_observer'

        if add_ray_noise:
            print("Using python avoidance policy, as we are unable to add noise otherwise")
            cpp_policy=False # Can't add noise with the cpp policy (or at least not in a straightforward way)

            builder.register_observer(lambda **planning_time_params: RayNoiser(**ray_noise_params, **planning_time_params), 
                                      observer_name="ray_noiser", 
                                      additional_params=[(ray_observer_name, "ray_observation_getter")])
            ray_observer_name = "ray_noiser"

        builder.add_policy(lambda **planning_time_params: LearnedRayPolicy(**learned_policy_params, **planning_time_params), 
                           ['target', 'observation_callback', (ray_observer_name, 'ray_observation_getter')], 
                           intercept=True, interceptor_name="learned_policy_output")

        if cpp_policy:
            builder.add_policy(lambda **planning_time_params: RaycastingCudaPolicy(**raycasting_cuda_policy_params.__dict__, **planning_time_params), 
                            ['tsdf'], intercept=True, interceptor_name="rays_avoidance")
        else:
            params = {"params": raycasting_cuda_policy_params}
            builder.add_policy(lambda **planning_time_params: RaysAvoidancePolicy(**params, **planning_time_params),
                            [(ray_observer_name, 'ray_observation_getter')], intercept=True, interceptor_name="rays_avoidance")


        return builder.build()

    @staticmethod
    def learned_planner_with_ray_observer(
        model: torch.nn.Module,
        raycasting_cuda_policy_params: RaycastingCudaPolicyParameters, 
        learned_policy_rmp_params: LearnedPolicyRmpParameters, 
        ray_observer_params: RayObserverParameters,
        planner_params: PlannerParameters,
        track_lstm_output_difference: bool = False,
        cpp_policy: bool = True,
        partial_observability_kwargs: Dict[str, Any]={},
        ):
        """Generate a planner with learned target policy. This planner saves intermediate ray observations
        """

        learned_policy_params = {
            "model": model,
            "learned_policy_rmp_params": learned_policy_rmp_params, 
            "track_lstm_output_difference": track_lstm_output_difference,
        }

        builder = PlannerBuilder(planner_params)
        
        ray_observer_name = PlannerFactory.setup_ray_observer(builder, ray_observer_params, **partial_observability_kwargs)

        builder.add_policy(lambda **planning_time_params: LearnedRayPolicy(**learned_policy_params, **planning_time_params), 
                           ['target', 'observation_callback', ('ray_observer', 'ray_observation_getter')], 
                           intercept=True, interceptor_name="learned_policy_output")
        
        if cpp_policy:
            builder.add_policy(lambda **planning_time_params: RaycastingCudaPolicy(**raycasting_cuda_policy_params.__dict__, **planning_time_params), 
                            ['tsdf'], intercept=True, interceptor_name="rays_avoidance")
        else:
            params = {"params": raycasting_cuda_policy_params}
            builder.add_policy(lambda **planning_time_params: RaysAvoidancePolicy(**params, **planning_time_params),
                            [(ray_observer_name, 'ray_observation_getter')], intercept=True, interceptor_name="rays_avoidance")


        return builder.build()
    
    @staticmethod
    def learned_labeled_from_path(
        path: str, 
        model_args: Dict[str, Any],
        *args, **kwargs
        ):
        """Generate a planner with learned target policy. 
        """
        model = RayModel(**copy.deepcopy(model_args))
        model.load_state_dict(torch.load(path))
        model.eval()
        model.to(torch.device("cuda"))
        model = RayModelDirectionConversionWrapper(model)

        return PlannerFactory.learned_labeled(model, *args, **kwargs)

    @staticmethod
    def learned_labeled(
        model, 
        raycasting_cuda_policy_params: RaycastingCudaPolicyParameters, 
        learned_policy_rmp_params: LearnedPolicyRmpParameters, 
        ray_observer_params: RayObserverParameters,
        planner_params: PlannerParameters,
        track_lstm_output_difference: bool = False,
        cpp_policy=True,
        partial_observability_kwargs: Dict[str, Any]={},
        add_ray_noise=False,
        ray_noise_params={},
        ):
        """Generate a planner with learned target policy. This planner saves intermediate ray observations, 
        and also adds an inactive expert policy that saves the geodesic field as a possible label. 
        """
        if cpp_policy and ("partial_ray_observability" in partial_observability_kwargs and partial_observability_kwargs["partial_ray_observability"] \
                           or add_ray_noise):
            raise ValueError("Partial ray observability or adding ray noise is not supported with cpp policy")
        

        learned_policy_params = {
            "model": model,
            "learned_policy_rmp_params": learned_policy_rmp_params, 
            "track_lstm_output_difference": track_lstm_output_difference,
        }

        # The expert policy uses these, but because it is inactive it does not matter
        target_policy_params = TargetPolicyParameters.from_yaml_general_config()
        expert_params = {
            "target_policy_rmp_params": target_policy_params,
        }

        builder = PlannerBuilder(planner_params)
        
        ray_observer_name = PlannerFactory.setup_ray_observer(builder, ray_observer_params,
                                                              ray_noise=add_ray_noise, ray_noise_params=ray_noise_params, 
                                                              **partial_observability_kwargs)

        builder.add_policy(lambda **planning_time_params: ExpertPolicy(**expert_params, **planning_time_params), 
                           ['target', 'geodesic', 'observation_callback'], active=False)
        builder.add_policy(lambda **planning_time_params: LearnedRayPolicy(**learned_policy_params, **planning_time_params), 
                           ['target', 'observation_callback', (ray_observer_name, 'ray_observation_getter')], 
                           intercept=True, interceptor_name="learned_policy_output")

        if cpp_policy:
            builder.add_policy(lambda **planning_time_params: RaycastingCudaPolicy(**raycasting_cuda_policy_params.__dict__, **planning_time_params), 
                            ['tsdf'], intercept=True, interceptor_name="rays_avoidance")
        else:
            params = {"params": raycasting_cuda_policy_params}
            builder.add_policy(lambda **planning_time_params: RaysAvoidancePolicy(**params, **planning_time_params),
                            [(ray_observer_name, 'ray_observation_getter')], intercept=True, interceptor_name="rays_avoidance")

        return builder.build()

        
    @staticmethod
    def expert_planner(
        target_policy_params: TargetPolicyParameters, 
        raycasting_cuda_policy_params: RaycastingCudaPolicyParameters, 
        planner_params: PlannerParameters,
    ):
        
        expert_params = {
            "target_policy_rmp_params": target_policy_params,
        }

        builder = PlannerBuilder(planner_params)

        builder.add_policy(lambda **planning_time_params: ExpertPolicy(**expert_params, **planning_time_params), 
                           ['target', 'geodesic', 'observation_callback'], intercept=True, interceptor_name="expert_interceptor")
        
        builder.add_policy(lambda **planning_time_params: RaycastingCudaPolicy(**raycasting_cuda_policy_params.__dict__, **planning_time_params), 
                           ['tsdf'], intercept=True, interceptor_name="rays_avoidance")
        
        return builder.build()
    

 

    @staticmethod
    def expert_labeled(
        target_policy_rmp_params: TargetPolicyParameters,
        raycasting_cuda_policy_params: RaycastingCudaPolicyParameters,
        ray_observer_params: RayObserverParameters,
        planner_params: PlannerParameters,
    ):
        expert_params = {
            "target_policy_rmp_params": target_policy_rmp_params,
        }
        
        ray_observer_params_policy = {
            "ray_observer_params": ray_observer_params
        }
        
        builder = PlannerBuilder(planner_params)
        builder.add_policy(lambda **planning_time_params: ExpertPolicy(**expert_params, **planning_time_params), 
                           ['target', 'geodesic', 'observation_callback'])
        builder.add_policy(lambda **planning_time_params: RaycastingCudaPolicy(**raycasting_cuda_policy_params.__dict__, **planning_time_params), 
                           ['tsdf'])
        
        builder.register_observer(lambda **planning_time_params: RayObserver(**ray_observer_params_policy, **planning_time_params), 
                                  observer_name="ray_observer",
                                  additional_params=['tsdf', 'observation_callback'])
        
        return builder.build()

    
    @staticmethod
    def baseline(
        target_policy_params: TargetPolicyParameters, 
        raycasting_cuda_policy_params: RaycastingCudaPolicyParameters, 
        planner_params: PlannerParameters
        ):
        """Generate a planner with a raycasting obstacle avoidance policy
        """

        builder = PlannerBuilder(planner_params)
        builder.add_policy(lambda **planning_time_params: SimpleTargetPolicy(**target_policy_params.__dict__, **planning_time_params), 
                           ['target'])

        builder.add_policy(lambda **planning_time_params: RaycastingCudaPolicy(**raycasting_cuda_policy_params.__dict__, **planning_time_params), 
                           ['tsdf'])
            
        return builder.build()

    @staticmethod
    def baseline_shortdistance_ray_observer(
        target_policy_params: TargetPolicyParameters,
        raycasting_cuda_policy_params: RaycastingCudaPolicyParameters,
        ray_observer_params: RayObserverParameters,
        planner_params: PlannerParameters,
        cpp_policy=True,
        save_rays=False,
    ):
        ray_observer_params_policy = {
            "ray_observer_params": ray_observer_params
        }

        builder = PlannerBuilder(planner_params)
        builder.add_policy(lambda **planning_time_params: SimpleTargetPolicy(**target_policy_params.__dict__, **planning_time_params),
                           ['target'], intercept=True, interceptor_name="simple_target")


        builder.register_observer(lambda **planning_time_params: RayObserver(**ray_observer_params_policy, **planning_time_params), 
                                observer_name="ray_observer",
                                additional_params=['tsdf'] + list(['observation_callback'] if save_rays else []))

        if cpp_policy:
            builder.add_policy(lambda **planning_time_params: RaycastingCudaPolicy(**raycasting_cuda_policy_params.__dict__, **planning_time_params), 
                            ['tsdf'], intercept=True, interceptor_name="rays_avoidance")
        else:
            params = {"params": raycasting_cuda_policy_params}
            builder.add_policy(lambda **planning_time_params: RaysAvoidancePolicy(**params, **planning_time_params),
                            [("ray_observer", 'ray_observation_getter')], intercept=True, interceptor_name="rays_avoidance")

        return builder.build()


    @staticmethod
    def setup_ray_observer(builder: PlannerBuilder, ray_observer_params: RayObserverParameters, 
                           partial_observability=False, 
                           forward_sensor_fov_deg=90,
                           kernel_size=5, save_intermediate_rays_every_step=1, smoothing_iterations=0, 
                           downsample_fraction=1.0, 
                           accumulator_steps=0, 
                           ray_noise=False, ray_noise_params={}) -> str:
        ray_observer_params_policy = {
            "ray_observer_params": ray_observer_params
        }
        # Ray observer that is used to get the ray observations that can be shared between the policies
        builder.register_observer(lambda **planning_time_params: RayObserver(**ray_observer_params_policy, **planning_time_params), 
                                  observer_name="ray_observer",
                                  additional_params=['tsdf', 'observation_callback'])
        
        ray_observer_name = "ray_observer"

        if ray_noise:
            builder.register_observer(lambda **planning_time_params: RayNoiser(**ray_noise_params, **planning_time_params), 
                                      observer_name="ray_noiser", 
                                      additional_params=['observation_callback', (ray_observer_name, "ray_observation_getter")])
            ray_observer_name = "ray_noiser"

        if not partial_observability:
            return ray_observer_name # We return the name of the observer so that it can be used in the policies

        downsampler_params = {
            "fraction": downsample_fraction
        }

        # Directions of the laser range finders (everything but forward)
        directions = [
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([-1, 0, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, -1]),
        ]

        # Laser sensors
        for i, direction in enumerate(directions):
            laser_ray_observer_params = {
                "num_rays": ray_observer_params.N_sqrt**2,
                "sensor_direction": direction, 
                "fov": 0,
            }
            # It is important that we capture the parameters here, otherwise the last value of the loop will be used
            # So that's why we use the default params=partial_ray_observer_forward_params in the lambda, so that the 
            # parameters are captured at the time of the loop, instead of capturing the reference to the dict which will 
            # be updated in the next iteration
            builder.register_observer(lambda params=laser_ray_observer_params, **planning_time_params: 
                                        PartialRayObserver(**params, **planning_time_params),
                                    observer_name=f"laser_partial_ray_observer_{i}", 
                                    additional_params=['target', (ray_observer_name, 'ray_observation_getter')],)

        
        builder.register_observer(lambda **planning_time_params: 
                                  RayStochasticDownsampler(**downsampler_params, **planning_time_params),
                                  observer_name="ray_stochastic_downsampler",
                                  additional_params=[(ray_observer_name, 'ray_observation_getter')])
        
        builder.register_observer(lambda **planning_time_params: RaySaver(name="stochastic_rays", **planning_time_params),
                                  additional_params=['observation_callback', ('ray_stochastic_downsampler', 'ray_observation_getter')])

        # Radar sensor forward params
        forward_partial_ray_observer_params = {
            "num_rays": ray_observer_params.N_sqrt**2,
            "sensor_direction": np.array([1, 0, 0]), 
            "fov": np.deg2rad(forward_sensor_fov_deg),
        }

        forward_ray_observer_name = "forward_partial_ray_observer"
        builder.register_observer(lambda params=forward_partial_ray_observer_params, **planning_time_params: 
                                  PartialRayObserver(**params, **planning_time_params),
                                  observer_name=forward_ray_observer_name,
                                  additional_params=['target', ('ray_stochastic_downsampler', 'ray_observation_getter')])
         
        if accumulator_steps > 0:
            forward_ray_observer_name = "ray_accumulator"
            builder.register_observer(lambda **planning_time_params: RayAccumulator(**planning_time_params, steps=accumulator_steps), 
                                    observer_name=forward_ray_observer_name, 
                                    additional_params=[("forward_partial_ray_observer", "ray_observation_getter")])


        builder.register_observer(lambda **planning_time_params: RayObservationCombiner(**planning_time_params),
                                  observer_name="ray_observation_combiner",
                                  additional_params=
                                  [f"laser_partial_ray_observer_{i}" for i in range(len(directions))] + 
                                  [forward_ray_observer_name]
                                  )

        builder.register_observer(lambda **planning_time_params: RaySaver(name="sensor_rays", **planning_time_params),
                                  additional_params=['observation_callback', ('ray_observation_combiner', 'ray_observation_getter')])
        
        interpolator_params = {
            "num_rays": ray_observer_params.N_sqrt**2,
            "kernel_size": kernel_size,
            "save_intermediate_rays": save_intermediate_rays_every_step,
            "iterations": smoothing_iterations,
            "callback_name": "ray_observation_interpolator",
        }
        
        builder.register_observer(lambda **planning_time_params: RayObservationInterpolator(**interpolator_params, **planning_time_params),
                                  observer_name="ray_observation_interpolator",
                                  additional_params=['observation_callback', ('ray_observation_combiner', 'ray_observation_getter')])

        builder.register_observer(lambda **planning_time_params: RaySaver(name="forward_rays", **planning_time_params),
                                  additional_params=['observation_callback', ('ray_observation_interpolator', 'ray_observation_getter')])

        return "ray_observation_interpolator"


