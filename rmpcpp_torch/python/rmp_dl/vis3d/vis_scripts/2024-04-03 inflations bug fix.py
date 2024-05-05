import os
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
import numpy as np
from rmp_dl.learning.data.pipeline.nodes.worlds.probabilistic_world import ProbabilisticWorld
from rmp_dl.learning.model import RayModelDirectionConversionWrapper, RayModelStaticSizeWrapper
from rmp_dl.learning.model_io.exporter import ModelExporter
from rmp_dl.vis3d.dash.dash_gui import DashApp
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import torch

import rmp_dl.util.io as rmp_io


def sizes():
    # worldgen = CustomWorldgenFactory.SlitWall()
    # worldgen = CustomWorldgenFactory.MultipleSlitWall()
    worldgen = CustomWorldgenFactory.WallDensities()
    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
    planner3dvis = Planner3dVis(worldgen, distancefield=False, num_arrows=8e3, distancefield_inflation=0.4)
    
    model = ModelUtil.load_model("nm0mws75", "last")
    # model = ModelUtil.load_model("5msibfu3", "latest")
    # model.model.set_maximum_ray_length(50)
    
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
        
    sizes = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    intensities = list(np.array(range(len(sizes))) / len(sizes))
    colors = Open3dUtils.intensities_to_rgb_with_mpl(intensities, colormap="viridis")
    for size, color in zip(sizes, colors):
        model_wrapped = RayModelStaticSizeWrapper(model, size)
        

        Planner3dVisFactory.add_learned_planner(planner3dvis, name=f"Radius - {size}",
                                                model=model_wrapped, color=color, 
                                                step_vis=False, ray_vis=True)
    
    # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 1, 0])

    planner3dvis.go()

def sizes2():
    # worldgen = CustomWorldgenFactory.SlitWall()
    # worldgen = CustomWorldgenFactory.MultipleSlitWall()
    # worldgen = CustomWorldgenFactory.WallDensities()
    worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=1421)
    planner3dvis = Planner3dVis(worldgen, distancefield=False, num_arrows=8e3, distancefield_inflation=0.2)
    
    model = ModelUtil.load_model("nm0mws75", "last")
    # model = ModelUtil.load_model("5msibfu3", "latest")
    # model.model.set_maximum_ray_length(50)
    
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
        
    sizes = [0.0, 0.05, 0.1, 0.2, 0.4]
    intensities = list(np.array(range(len(sizes))) / len(sizes))
    colors = Open3dUtils.intensities_to_rgb_with_mpl(intensities, colormap="viridis")
    for size, color in zip(sizes, colors):
        model_wrapped = RayModelStaticSizeWrapper(model, size)
        

        Planner3dVisFactory.add_learned_planner(planner3dvis, name=f"Radius - {size}",
                                                model=model_wrapped, color=color, 
                                                step_vis=False, ray_vis=True)
    
    # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 1, 0])

    planner3dvis.go()
    

        

if __name__ == "__main__":
    DashApp.run()
    sizes()
    # rnn()

