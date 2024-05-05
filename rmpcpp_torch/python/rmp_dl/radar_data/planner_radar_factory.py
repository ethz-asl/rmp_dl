
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.planner.observers.ray_observation_interpolator import RayObservationInterpolator
from rmp_dl.planner.observers.ray_saver import RaySaver
from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, RaycastingCudaPolicyParameters
from rmp_dl.planner.policies.learned_ray_policy import LearnedRayPolicy
from rmp_dl.planner.policies.rays_avoidance_policy import RaysAvoidancePolicy
from rmp_dl.radar_data.observers.radar_points_accumulator import RadarPointsAccumulator
from rmp_dl.radar_data.observers.radar_points_getter import RadarPointsGetter
from rmp_dl.radar_data.observers.radar_points_ray_converter import RadarPointsRayConverter
from rmp_dl.radar_data.planner_radar_builder import PlannerRadarBuilder


class PlannerRadarFactory:
    @staticmethod
    def radar_planner(
        model: RayModelDirectionConversionWrapper,
        acc_steps: int = 10, 
        ):
        builder = PlannerRadarBuilder()
        builder.register_observer(lambda **planning_time_params: RadarPointsGetter(**planning_time_params), 
                                  "radar_points", 
                                  ["radar_data", "observation_callback"])

        
        builder.register_observer(lambda **planning_time_params: RadarPointsAccumulator(**planning_time_params, steps=acc_steps), 
                                  "radar_points_accumulator", 
                                  [("radar_points", "observation_getter")])
        
        builder.register_observer(lambda **planning_time_params: RadarPointsRayConverter(**planning_time_params, num_rays=1024),
                                  "radar_points_ray_converter",
                                  [("radar_points_accumulator", "observation_getter")]
                                  )
        
        builder.register_observer(lambda **planning_time_params: RaySaver(**planning_time_params, name="radar_converter"),
                                  additional_params=[("radar_points_ray_converter", "ray_observation_getter"), "observation_callback"])


        builder.register_observer(lambda **planning_time_params: RayObservationInterpolator(**planning_time_params, num_rays=1024),
                                  "ray_interpolator", 
                                  [("radar_points_ray_converter", "ray_observation_getter")])
        
        builder.register_observer(lambda **planning_time_params: RaySaver(**planning_time_params, name="ray_interpolator"),
                                  additional_params=[("ray_interpolator", "ray_observation_getter"), "observation_callback"])


        learned_policy_params = {
            "model": model,
            "learned_policy_rmp_params": LearnedPolicyRmpParameters.from_yaml_general_config(),
            "target": None, # This means that we use the velocity direction as the target
        }
        builder.add_policy(lambda **planning_time_params: LearnedRayPolicy(**learned_policy_params, **planning_time_params), 
                            [("ray_interpolator", "ray_observation_getter"), "observation_callback"], 
                            intercept=True, 
                            interceptor_name="learned_policy_interceptor")


        avoidance_params = {
            "params": RaycastingCudaPolicyParameters.from_yaml_general_config(),
        }
        builder.add_policy(lambda **planning_time_params: RaysAvoidancePolicy(**planning_time_params, **avoidance_params), 
                           [("ray_interpolator", "ray_observation_getter")], 
                           intercept=True, 
                           interceptor_name="avoidance_policies")

        return builder.build()