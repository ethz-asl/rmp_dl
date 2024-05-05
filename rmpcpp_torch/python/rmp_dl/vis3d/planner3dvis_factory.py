
import collections
from typing import Callable, List, Optional
import numpy as np
from rmp_dl.learning.lightning_module import RayLightningModule
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, PlannerParameters, RayObserverParameters, RaycastingCudaPolicyParameters, TargetPolicyParameters
from rmp_dl.util.io import resolve_directory
from rmp_dl.vis3d.arrow import get_arrow_geometry
from rmp_dl.vis3d.dash.dash_gui import DashApp, VerticalParamList
from rmp_dl.vis3d.geodesic_vis import DenseSampledVis
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.vis3d.vis3d import KeyModifier, Plot3D
import torch
import wandb

import torch.nn as nn


def additive_parameter_change(obj, key, increment, ctrl_mult=5, alt_mult=20):
    def change_parameter(mods, increment=increment):
        if mods & KeyModifier.SHIFT: 
            increment = -increment
        if mods & KeyModifier.CTRL:
            increment = increment * ctrl_mult
        if mods & KeyModifier.ALT:
            increment = increment * alt_mult
        
        obj.__setattr__(key, obj.__getattribute__(key) + increment)
    
    return change_parameter

def multiplicative_parameter_change(obj, key, increment, ctrl_mult=8, alt_mult=64):
    def change_parameter(mods, increment=increment):
        if mods & KeyModifier.CTRL:
            increment = increment * ctrl_mult
        if mods & KeyModifier.ALT:
            increment = increment * alt_mult
        if mods & KeyModifier.SHIFT: 
            increment = 1/increment
        
        obj.__setattr__(key, obj.__getattribute__(key) * increment)
    
    return change_parameter

def boolean_parameter_change(obj, key):
    def change_parameter(mods):
        obj.__setattr__(key, not obj.__getattribute__(key))
    
    return change_parameter

class Planner3dVisFactory:
    @staticmethod
    def add_simple_target_planner(planner3dvis, color=None, step_vis=False, ray_vis=False,
                                  cpp_policy=True,
                                  trajectory_size=None, name="simple_target_planner", 
                                  raycasting_cuda_policy_params_name="raycasting_avoidance", 
                                  key_replan=True):
        constructor = PlannerFactory.baseline_shortdistance_ray_observer

        target_policy_params = TargetPolicyParameters.from_yaml_general_config()
        raycasting_cuda_policy_params = RaycastingCudaPolicyParameters.from_yaml_general_config(name=raycasting_cuda_policy_params_name)
        ray_observer_params = RayObserverParameters.from_yaml_general_config()
        ray_observer_params.maximum_ray_length = raycasting_cuda_policy_params.r
        planner_params = PlannerParameters.from_yaml_general_config()

        planner_constructor_params = {
            "target_policy_params": target_policy_params,
            "raycasting_cuda_policy_params": raycasting_cuda_policy_params,
            "ray_observer_params": ray_observer_params,
            "planner_params": planner_params,
            "cpp_policy": cpp_policy,
            "save_rays": ray_vis
        }

        key_replan_parameter_changes = []
        if key_replan:
            key_replan_parameter_changes = [
                (ord("A"), additive_parameter_change(target_policy_params, "alpha", 1.0)),
                (ord("B"), additive_parameter_change(target_policy_params, "beta", 1.0)),
                (ord("E"), additive_parameter_change(raycasting_cuda_policy_params, "eta_rep", 1.0)),
                (ord("D"), additive_parameter_change(raycasting_cuda_policy_params, "eta_damp", 1.0)),
                (ord("R"), additive_parameter_change(raycasting_cuda_policy_params, "v_rep", 0.1)),
                (ord("F"), additive_parameter_change(raycasting_cuda_policy_params, "v_damp", 0.1)),


                (ord("I"), boolean_parameter_change(raycasting_cuda_policy_params, "metric")),

                (ord("M"), multiplicative_parameter_change(raycasting_cuda_policy_params, "metric_scale", 2)),
                (Plot3D.COMMA, multiplicative_parameter_change(raycasting_cuda_policy_params, "force_scale", 2)),
            ]

        key_replan_parameter_changes = Planner3dVisFactory.add_replan_parameter_intercept_callbacks(
            name, key_replan_parameter_changes, target_policy_params, raycasting_cuda_policy_params)

        planner3dvis.add_planner(name, constructor, planner_constructor_params, key_replan_parameter_changes, color=color, trajectory_size=trajectory_size)
        
        planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_pos_visualization_callback(name))

        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_print_matrix_with_eigenvectors_callback(
            name, lambda d: d["rays_avoidance"]["A"], "Avoidance Metric"
        )) 
        
        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_print_matrix_callback(
            name, lambda d: d["rays_avoidance"]["f"], "Avoidance Force"
        )) 
 
        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_print_matrix_callback(
            name, lambda d: d["simple_target"]["f"], "Target Force"
        )) 

        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_intermediate_rmp_results_callback(name))

        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_state_callback(name))

        if ray_vis:
            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(
                name, rays_name="rays", mod_idx=planner3dvis.increment_and_get_step_geometry_idx())) 


    @staticmethod
    def add_learned_planner(planner3dvis, model, color=None, name=None, 
                            ray_vis=False, step_vis=False, lstm_color=False, 
                            use_meshes=False, mesh_radius=0.03,
                            ray_pred_size=1.0, 
                            step_vis_choices: Optional[set] = None, 
                            trajectory_size=None, 
                            cpp_policy=True, 
                            add_ray_noise=False, 
                            ray_noise_params={},
                            partial_observability_kwargs={},
                            ):
        raycasting_cuda_policy_params = RaycastingCudaPolicyParameters.from_yaml_general_config()
        learned_policy_params = LearnedPolicyRmpParameters.from_yaml_general_config()
        ray_observer_params = RayObserverParameters.from_yaml_general_config()
        planner_params = PlannerParameters.from_yaml_general_config()
        
        if name is None: 
            name = str(hash(model))

        constructor = PlannerFactory.learned_labeled
        key_replan_parameter_changes = [
            # ("A", lambda: learned_policy_params.alpha, lambda x: learned_policy_params.__setattr__("alpha", x), 1.0),
            # ("B", lambda: learned_policy_params.beta, lambda x: learned_policy_params.__setattr__("beta", x), 1.0),
            # ("C", lambda: learned_policy_params.c_softmax, lambda x: learned_policy_params.__setattr__("c_softmax", x), 0.01),
            ]
        
        model.eval()

        planner_params_full = {
            "model": model,
            "raycasting_cuda_policy_params": raycasting_cuda_policy_params,
            "learned_policy_rmp_params": learned_policy_params,
            "ray_observer_params": ray_observer_params,
            "planner_params": planner_params, 
            "cpp_policy": cpp_policy,
            "partial_observability_kwargs": partial_observability_kwargs,
            "add_ray_noise": add_ray_noise,
            "ray_noise_params": ray_noise_params,
        }
        
        # If we plot the color of how an lstm changes the output, we also track the output difference.
        if lstm_color:
            planner_params_full["track_lstm_output_difference"] = True

        if lstm_color:
            color = Planner3dVisFactory.lstm_color_callback(name)

        planner3dvis.add_planner(
            name, constructor, planner_params_full, key_replan_parameter_changes, color=color, trajectory_size=trajectory_size)


        if step_vis:
            if step_vis_choices is None or "learned" in step_vis_choices:
                planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_arrow_visualization_callback(
                    name, 
                    get_direction_from_dict_callable=lambda d: d["learned_policy"]["geodesic_prediction"], 
                    color=[1, 0, 0])
                    )

            if step_vis_choices is None or "goal" in step_vis_choices:
                def get_relative_position(d: dict):
                    rel_pos = d["learned_policy"]["goal"] - d["state"]["pos"]
                    return rel_pos / np.linalg.norm(rel_pos)
                
                planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_arrow_visualization_callback(
                    name, 
                    get_direction_from_dict_callable=get_relative_position, 
                    color=[0, 1, 0])
                    )
            
            if step_vis_choices is None or "geodesic" in step_vis_choices:
                planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_arrow_visualization_callback(
                    name, 
                    get_direction_from_dict_callable=lambda d: d["expert_policy"]["geodesic"], 
                    color=[0, 0, 0])
                    )
        
        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_print_matrix_with_eigenvectors_callback(
            name, lambda d: d["rays_avoidance"]["A"], "Avoidance Metric"
        )) 
        
        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_print_matrix_callback(
            name, lambda d: d["rays_avoidance"]["f"], "Avoidance Force"
        )) 
 
        planner3dvis.register_step_geometry_change(Planner3dVisFactory.get_print_matrix_callback(
            name, lambda d: d["learned_policy_output"]["f"], "Learned Force"
        )) 
        
        if ray_vis:
            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_pred_visualization_callback(
                name, mod_idx=planner3dvis.increment_and_get_step_geometry_idx(), 
                use_meshes=use_meshes, size=ray_pred_size, mesh_radius=mesh_radius))

            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(
                name, rays_name="rays", mod_idx=planner3dvis.increment_and_get_step_geometry_idx())) 

            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(
                name, rays_name="rays_noisy", mod_idx=planner3dvis.increment_and_get_step_geometry_idx()))

            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(
                name, rays_name="stochastic_rays", mod_idx=planner3dvis.increment_and_get_step_geometry_idx()))

            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(
                name, rays_name="sensor_rays", mod_idx=planner3dvis.increment_and_get_step_geometry_idx()))
            
            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_list_visualization_callback(
                name, rays_name="ray_observation_interpolator", length=5, planner3dvis=planner3dvis))


    @staticmethod
    def add_learned_planner_minimal_with_ray_observer(planner3dvis, model, color=None, name=None, 
                            ray_vis=False, step_vis=False, lstm_color=False, 
                            use_meshes=False, mesh_radius=0.03,
                            ray_pred_size=1.0, 
                            step_vis_choices: Optional[set] = None, 
                            trajectory_size=None, 
                            cpp_policy=True):
        raycasting_cuda_policy_params = RaycastingCudaPolicyParameters.from_yaml_general_config()
        learned_policy_params = LearnedPolicyRmpParameters.from_yaml_general_config()
        ray_observer_params = RayObserverParameters.from_yaml_general_config()
        planner_params = PlannerParameters.from_yaml_general_config()
        
        if name is None: 
            name = str(hash(model))

        constructor = PlannerFactory.learned_planner_with_ray_observer
        key_replan_parameter_changes = [
            # ("A", lambda: learned_policy_params.alpha, lambda x: learned_policy_params.__setattr__("alpha", x), 1.0),
            # ("B", lambda: learned_policy_params.beta, lambda x: learned_policy_params.__setattr__("beta", x), 1.0),
            # ("C", lambda: learned_policy_params.c_softmax, lambda x: learned_policy_params.__setattr__("c_softmax", x), 0.01),
            ]
        
        model.eval()

        planner_params_full = {
            "model": model,
            "raycasting_cuda_policy_params": raycasting_cuda_policy_params,
            "learned_policy_rmp_params": learned_policy_params,
            "ray_observer_params": ray_observer_params,
            "planner_params": planner_params, 
            "cpp_policy": cpp_policy,
        }
        
        # If we plot the color of how an lstm changes the output, we also track the output difference.
        if lstm_color:
            planner_params_full["track_lstm_output_difference"] = True

        if lstm_color:
            color = Planner3dVisFactory.lstm_color_callback(name)

        planner3dvis.add_planner(
            name, constructor, planner_params_full, key_replan_parameter_changes, color=color, trajectory_size=trajectory_size)

        if step_vis:
            if step_vis_choices is None or "learned" in step_vis_choices:
                planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_arrow_visualization_callback(
                    name, 
                    get_direction_from_dict_callable=lambda d: d["learned_policy"]["geodesic_prediction"], 
                    color=[1, 0, 0])
                    )

            if step_vis_choices is None or "goal" in step_vis_choices:
                def get_relative_position(d: dict):
                    rel_pos = d["learned_policy"]["goal"] - d["state"]["pos"]
                    return rel_pos / np.linalg.norm(rel_pos)
                
                planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_arrow_visualization_callback(
                    name, 
                    get_direction_from_dict_callable=get_relative_position, 
                    color=[0, 1, 0])
                    )
            
        if ray_vis:
            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_visualization_callback(name, mod_idx=planner3dvis.increment_and_get_step_geometry_idx()))
            planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_ray_pred_visualization_callback(
                name, use_meshes=use_meshes, size=ray_pred_size, mesh_radius=mesh_radius, mod_idx=planner3dvis.increment_and_get_step_geometry_idx()))

    @staticmethod
    def add_learned_planner_minimal(planner3dvis, model, color=None, name=None, 
                            trajectory_size=None, ):
        raycasting_cuda_policy_params: dict = RaycastingCudaPolicyParameters.from_yaml_general_config()
        learned_policy_params: dict = LearnedPolicyRmpParameters.from_yaml_general_config()
        
        if name is None: 
            name = str(hash(model))

        constructor = PlannerFactory.learned_planner_minimal
        key_replan_parameter_changes = [
            # ("A", lambda: learned_policy_params["alpha"], lambda x: learned_policy_params.update({"alpha": x}), 1.0),
            # ("B", lambda: learned_policy_params["beta"], lambda x: learned_policy_params.update({"beta": x}), 1.0),
            # ("C", lambda: learned_policy_params["c_softmax"], lambda x: learned_policy_params.update({"c_softmax": x}), 0.01),
            ]
        
        if issubclass(type(model), nn.Module):
            model.eval()

        ray_observer_params = RayObserverParameters.from_yaml_general_config()
        planner_params = PlannerParameters.from_yaml_general_config()

        planner_params_full = {
            "model": model,
            "raycasting_cuda_policy_params": raycasting_cuda_policy_params,
            "learned_policy_rmp_params": learned_policy_params,
            "ray_observer_params": ray_observer_params,
            "planner_params": planner_params,
        }

        planner3dvis.add_planner(
            name, constructor, planner_params_full, key_replan_parameter_changes, color=color, trajectory_size=trajectory_size)

    
    @staticmethod
    def add_expert_planner(planner3dvis, worldgen):
        name = "expert"
        constructor = PlannerFactory.expert_labeled

        target_policy_params: dict = TargetPolicyParameters.from_yaml_general_config().__dict__.copy()
        raycasting_cuda_policy_params: dict = RaycastingCudaPolicyParameters.from_yaml_general_config(expert=True).__dict__.copy()

        api = wandb.Api()
        file = api.artifact("rmp_dl/rmp_dl/model-lsjr67b4:latest").download(root=resolve_directory("data"))

        model = RayLightningModule.load_from_checkpoint(file + "/model.ckpt").to(torch.device("cuda"))
        learned_policy_params: dict = LearnedPolicyRmpParameters.from_yaml_general_config().__dict__.copy()

        callbacks = [
            ("A", lambda: target_policy_params["alpha"], lambda x: target_policy_params.update({"alpha": x}), 1.0),
            ("E", lambda: raycasting_cuda_policy_params["eta_rep"], lambda x: raycasting_cuda_policy_params.update({"eta_rep": x}), 10.0),
            ("R", lambda: raycasting_cuda_policy_params["eta_damp"], lambda x: raycasting_cuda_policy_params.update({"eta_damp": x}), 10.0),
            ("V", lambda: raycasting_cuda_policy_params["v_rep"], lambda x: raycasting_cuda_policy_params.update({"v_rep": x}), 0.1),
            ("B", lambda: raycasting_cuda_policy_params["v_damp"], lambda x: raycasting_cuda_policy_params.update({"v_damp": x}), 0.1),
        ]

        planner_params = {
            "model": model,
            "target_policy_params": target_policy_params,
            "raycasting_cuda_policy_params": raycasting_cuda_policy_params,
            "learned_policy_params": learned_policy_params, 
            "worldgen_settings": worldgen.get_settings(),
        }

        planner3dvis.add_planner(name, constructor, planner_params, callbacks)
        
        planner3dvis.register_step_geometry_change(Planner3dVisFactory.add_arrow_visualization_callback(
            name, 
            get_direction_from_dict_callable = lambda dic: dic["expert_policy"]["geodesic"]))
        
    @staticmethod
    def add_densely_sampled_data(planner3dvis, worldgen, num=1024, scale=0.2):
        DenseSampledVis(planner3dvis.plot3d, worldgen).plot_locations(num, scale=scale)
        
        
    @staticmethod
    def add_arrow_visualization_callback(name: str, 
                                                 get_direction_from_dict_callable: Callable[[dict], np.ndarray], 
                                                 color=[1, 0, 0]):
        def callback(observations: dict, idx: int, step_geometry_modifier: int):
            observations = observations[name]
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)
            d = get_direction_from_dict_callable(observations[idx])
            
            pos = observations[idx]["state"]["pos"]
            print(d)
            arrow = get_arrow_geometry(pos, d, scale=0.3)
            arrow.paint_uniform_color(color)
            return arrow    

        return callback

    @staticmethod
    def add_ray_list_visualization_callback(name: str, rays_name, length, planner3dvis):
        indices = [planner3dvis.increment_and_get_step_geometry_idx() for _ in range(length)]

        def callback(observations: dict, idx: int, step_geometry_modifier):
            if not (step_geometry_modifier in indices):
                return None
            
            observations = observations[name]
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)
            rays = observations[idx][rays_name]["ray_list"][step_geometry_modifier - indices[0]]

            pos = np.array(observations[idx]["state"]["pos"])
            return Open3dUtils.get_rays_geometry(rays, pos)

        def try_catch(observations: dict, idx: int, step_geometry_modifier):
            try:
                return callback(observations, idx, step_geometry_modifier)
            except:
                return None

        return try_catch


    @staticmethod
    def add_ray_visualization_callback(name: str, rays_name="rays", mod_idx=1):
        def callback(observations: dict, idx: int, step_geometry_modifier):
            if not (step_geometry_modifier == mod_idx):
                return None
            
            observations = observations[name]
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)
            try:
                rays = observations[idx][rays_name]["rays"]
            except (KeyError, IndexError): return []

            pos = np.array(observations[idx]["state"]["pos"])
            return Open3dUtils.get_rays_geometry(rays, pos)
            
        return callback

    @staticmethod
    def add_ray_pred_visualization_callback(name: str, mod_idx: int, use_meshes=False, size=1.0, mesh_radius=0.03):
        def callback(observations: dict, idx: int, step_geometry_modifier):
            if not (step_geometry_modifier == mod_idx):
                return None
            
            observations = observations[name]
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)

            def softmax_and_normalize(rays):
                rays = np.exp(rays)
                rays = rays / np.sum(rays)
                return rays

            try:
                rays = observations[idx]["learned_policy"]["output_ray_predictions"]
                rays = softmax_and_normalize(rays)
            except (KeyError, IndexError):
                return []
            
            # Normalize and resize
            rays /= np.max(rays)
            rays *= size

            pos = np.array(observations[idx]["state"]["pos"])

            # For the presentation the rays have to be a lot thicker. However, because of openGL version issues the easy way does not work. 
            # See: https://github.com/isl-org/Open3D/issues/1480 and https://github.com/isl-org/Open3D/pull/738 
            # Someone posted a workaround by using meshes instead of lines. However, this is not as performant as lines.
            if use_meshes:
                return Open3dUtils.get_rays_geometry_mesh(rays, pos, radius=mesh_radius)

            return Open3dUtils.get_rays_geometry(rays, pos)
            
        return callback

    @staticmethod
    def lstm_color_callback(name): 
        def callback(observations: dict) -> List[float]:
            colors = []
            observations = observations[name]
            colors = [sum(observation["learned_policy"]["recurrent_diff_norm"].values()) for observation in observations]
            colors.insert(0, 0.)  # First step is 0
            return colors
        
        return callback


    @staticmethod
    def add_full_trajectory_arrow_visualization_callback(
            name: str, 
            modifier_idx: int,
            get_direction_from_dict_callable: Callable[[dict], np.ndarray], 
            color=[1, 0, 0], 
            arrow_scale: float=0.05, 
            constant_scale = True
            ):
        def callback(observations: dict, global_geometry_modifier: int):
            if global_geometry_modifier != modifier_idx:
                return []

            observations = observations[name]
            
            try:
                directions = [get_direction_from_dict_callable(obs) for obs in observations]
            except IndexError:
                return
            positions = [obs["state"]["pos"] for obs in observations]
            if constant_scale is True:
                # We scale according to the norm of the vector
                scales = [np.linalg.norm(get_direction_from_dict_callable(obs)) for obs in observations]
                # We normalize between 0 and $scale
                max_scale = max(scales)
                scales = [scale / max_scale * arrow_scale for scale in scales]
            else:
                scales = [arrow_scale] * len(observations) 

            arrows = []
            for pos, d, scale in zip(positions, directions, scales):
                if np.linalg.norm(d) < 1e-6:
                    continue
                arrow = get_arrow_geometry(pos, d, scale=scale)
                arrow.paint_uniform_color(color)
                arrows.append(arrow)

            return arrows

        return callback
    

    @staticmethod
    def get_intermediate_rmp_results_callback(
        name: str
    ):
        def callback(observations: dict, idx: int, step_geometry_modifier: int):
            observations = observations[name]
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)

            A_avoidance = observations[idx]["rays_avoidance"]["A"]
            f_avoidance = observations[idx]["rays_avoidance"]["f"]
            A_target = observations[idx]["simple_target"]["A"]
            f_target = observations[idx]["simple_target"]["f"]
            
            data = []

            data.append("A_sum")
            data.append(A_avoidance + A_target)

            A_sum_pinv = np.linalg.pinv(A_avoidance + A_target)
            data.append("A_sum_pinv")
            data.append(A_sum_pinv)

            metric_x_force_sum = A_avoidance @ f_avoidance + A_target @ f_target
            data.append("metric_x_force_sum")
            data.append(metric_x_force_sum)

            data.append("final_force")
            data.append(A_sum_pinv @ metric_x_force_sum)

            DashApp.update_data(name, 'left', 'Intermediate RMP Results', data)
        return callback
    
    @staticmethod
    def get_state_callback(
        name: str
    ):
        def callback(observations: dict, idx: int, step_geometry_modifier: int):
            observations = observations[name]
            # We shift the state such that the acceleration lines up with the policy outputs
            idx = idx + 1 
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)

            pos = observations[idx]["state"]["pos"]
            vel =  observations[idx]["state"]["vel"]
            acc = observations[idx]["state"]["acc"]

            
            data = []

            data.append("pos")
            data.append(pos)
            data.append("vel")
            data.append(vel)
            data.append("acc")
            data.append(acc)

            DashApp.update_data(name, 'left', 'Next State', data)
        return callback



    @staticmethod
    def get_print_matrix_callback(
        name: str,
        get_matrix_from_dict_callable: Callable[[dict], np.ndarray],
        matrix_name: str,
    ):
        def callback(observations: dict, idx: int, step_geometry_modifier: int):
            observations = observations[name]
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)
            try:
                m = get_matrix_from_dict_callable(observations[idx])
            except IndexError:
                return

            DashApp.update_data(name, 'left', matrix_name, m)

        return callback

    @staticmethod
    def get_print_matrix_with_eigenvectors_callback(
        name: str,
        get_matrix_from_dict_callable: Callable[[dict], np.ndarray],
        matrix_name: str,
    ):
        def callback(observations: dict, idx: int, step_geometry_modifier: int):
            observations = observations[name]
            idx = min(len(observations) - 1, idx)
            idx = max(0, idx)
            try:
                m = get_matrix_from_dict_callable(observations[idx])
            except IndexError:
                return

            eigenvalues, eigenvectors = np.linalg.eig(m)
            
            # extract eigenvalues into list
            eigenvalues = eigenvalues.tolist()

            # extract each eigenvector to list
            eigenvectors = eigenvectors.T.tolist()

            # Sort the eigenvalues and eigenvectors according to eigenvalue size
            eigenvalue_vector_pair = sorted(zip(eigenvalues, eigenvectors), key=lambda x: x[0], reverse=True)

            data = []
            data.append(m)
            
            for i, (eigenvalue, eigenvector) in enumerate(eigenvalue_vector_pair):
                data.append(f"λ<sub>{i}</sub> = {eigenvalue:.2e}")
                data.append(np.array(eigenvector))

            DashApp.update_data(name, 'left', matrix_name, data)

        return callback


    @staticmethod
    def add_replan_parameter_intercept_callbacks(name, callback_tuples, *dataclasses):
        # We wrap the setter of the callback such that we send the new parameters to the dashboard. 
        # All dataclasses used in the getters and setters are used by reference (i.e. they live in the closure of the lambdas)
        # We get the same dataclasses as the ones in the lambda closures, which makes everything work
        # Slightly ugly but it is the easiest 
        def send_to_dash():
            for dataclass in dataclasses:
                data_class_name = type(dataclass).__name__
                params = VerticalParamList(dataclass.__dict__)
                DashApp.update_data(name, 'right', data_class_name, params)

        def setter_wrapper(setter):
            def wrapped_setter(value):
                setter(value)
                send_to_dash()
            return wrapped_setter
        
        for i in range(len(callback_tuples)):
            callback_tuple = callback_tuples[i]
            # Wrap the setter with an update to the dashboard
            callback_tuple = list(callback_tuple)
            callback_tuple[1] = setter_wrapper(callback_tuple[1])
            callback_tuple = tuple(callback_tuple)
            callback_tuples[i] = callback_tuple
        
        send_to_dash()

        return callback_tuples


    @staticmethod
    def add_pos_visualization_callback(name, radius=0.01):
        def callback(observations: dict, idx: int, step_geometry_modifier):
            try: 
                pos = observations[name][idx]["state"]["pos"]
            except IndexError:
                return []

            geom = Plot3D.get_sphere_geometry(pos, radius=radius, color=[0.0, 0.0, 0.0])

            return geom
        
        return callback