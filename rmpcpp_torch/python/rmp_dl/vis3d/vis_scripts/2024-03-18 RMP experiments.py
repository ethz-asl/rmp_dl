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
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import torch

import rmp_dl.util.io as rmp_io


def narrow_gaps():
    # worldgen = CustomWorldgenFactory.SlitWall()
    # worldgen = CustomWorldgenFactory.MultipleSlitWall()

    # worldgen = CustomWorldgenFactory.WallDensities()
    # worldgen = CustomWorldgenFactory.NarrowGaps()
    # worldgen = CustomWorldgenFactory.Overhang()
    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
    worldgen = CustomWorldgenFactory.BigSpheres()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, num_arrows=8e3, distancefield_inflation=0.0, 
                                voxel_grid=True, 
                                voxel_grid_loc = np.array([2.0, 6.0, 5.0]), 
                                voxel_grid_size = np.array([3.0, 3.0, 3.0]), 
                                )
    
    Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[1, 0, 0], cpp_policy=False, name="S Py", ray_vis=True, 
                                                  raycasting_cuda_policy_params_name="raycasting_avoidance2")
    
    Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 1, 0], cpp_policy=True, name="S cpp", ray_vis=False,
                                                  raycasting_cuda_policy_params_name="raycasting_avoidance", key_replan=False)

    planner3dvis.go()
        
if __name__ == "__main__":
    DashApp.run() 

    narrow_gaps()
