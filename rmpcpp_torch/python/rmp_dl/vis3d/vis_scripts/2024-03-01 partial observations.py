import os
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
import numpy as np
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.exporter import ModelExporter
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import torch

import rmp_dl.util.io as rmp_io


def rnn_over_wall():
    worldgen = CustomWorldgenFactory.OverhangY()

    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
    planner3dvis = Planner3dVis(worldgen, distancefield=False)


    model = ModelUtil.load_model("g2j8uxxd", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    
    # Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNNf",
    #                                         model=model, color=[1, 0, 0],
    #                                         step_vis=False, ray_vis=True, cpp_policy=False, )

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, color=[1, 0, 0],
                                            step_vis=False, ray_vis=False, cpp_policy=False)

    partial_observability_kwargs = {
        "partial_observability": True,
        "kernel_size": 5, 
        "save_intermediate_rays_every_step": 1,
        "smoothing_iterations": 0,
        "downsample_fraction": 1.0,
        "forward_sensor_fov_deg": 90,
    }

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN-f90",
                                            model=model, color=[1, 1, 0],
                                            step_vis=False, ray_vis=False, cpp_policy=False, 
                                            partial_observability_kwargs=partial_observability_kwargs)
    
    partial_observability_kwargs = {
        "partial_observability": True,
        "kernel_size": 5, 
        "save_intermediate_rays_every_step": 1,
        "smoothing_iterations": 0,
        "downsample_fraction": 1.0,
        "forward_sensor_fov_deg": 80,
    }

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN-f80",
                                            model=model, color=[0, 1, 0],
                                            step_vis=False, ray_vis=False, cpp_policy=False, 
                                            partial_observability_kwargs=partial_observability_kwargs)
    partial_observability_kwargs = {
        "partial_observability": True,
        "kernel_size": 5, 
        "save_intermediate_rays_every_step": 1,
        "smoothing_iterations": 0,
        "downsample_fraction": 1.0,
        "forward_sensor_fov_deg": 70,
    }

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN-f70",
                                            model=model, color=[0, 1, 1],
                                            step_vis=False, ray_vis=False, cpp_policy=False, 
                                            partial_observability_kwargs=partial_observability_kwargs)
    partial_observability_kwargs = {
        "partial_observability": True,
        "kernel_size": 5, 
        "save_intermediate_rays_every_step": 1,
        "smoothing_iterations": 0,
        "downsample_fraction": 1.0,
        "forward_sensor_fov_deg": 60,
    }

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN-f60",
                                            model=model, color=[0, 0, 1],
                                            step_vis=False, ray_vis=True, cpp_policy=False, 
                                            partial_observability_kwargs=partial_observability_kwargs)
    
    planner3dvis.go()


def rnn_over_wall_subs():
    worldgen = CustomWorldgenFactory.OverhangY()

    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
    planner3dvis = Planner3dVis(worldgen, distancefield=False)


    model = ModelUtil.load_model("g2j8uxxd", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    

    partial_observability_kwargs = {
        "partial_observability": True,
        "kernel_size": 5, 
        "save_intermediate_rays_every_step": 1,
        "smoothing_iterations": 0,
        "downsample_fraction": 0.03,
        "forward_sensor_fov_deg": 160,
    }

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN-f90",
                                            model=model, color=[1, 0, 0],
                                            step_vis=False, ray_vis=True, cpp_policy=False, 
                                            partial_observability_kwargs=partial_observability_kwargs)
    
    planner3dvis.go()

def rnn_over_wall_subs_acc():
    worldgen = CustomWorldgenFactory.OverhangY()

    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
    planner3dvis = Planner3dVis(worldgen, distancefield=False)


    model = ModelUtil.load_model("g2j8uxxd", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    

    partial_observability_kwargs = {
        "partial_observability": True,
        "kernel_size": 5, 
        "save_intermediate_rays_every_step": 1,
        "smoothing_iterations": 0,
        "downsample_fraction": 0.03,
        "forward_sensor_fov_deg": 160,
        "accumulator_steps": 5,
    }

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN-f90",
                                            model=model, color=[1, 0, 0],
                                            step_vis=False, ray_vis=True, cpp_policy=False, 
                                            partial_observability_kwargs=partial_observability_kwargs)
    
    planner3dvis.go()

if __name__ == "__main__":
    # rnn_over_wall()
    # rnn_over_wall_subs()
    rnn_over_wall_subs_acc()

