import copy
from comparison_planners.chomp.planner_chomp import ChompPlanner
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
from comparison_planners.rrt.planner_rrt import RRTPlanner
import numpy as np
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.importer import ModelImporter
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, PlannerParameters, RayObserverParameters, RaycastingCudaPolicyParameters, TargetPolicyParameters
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import torch

import rmp_dl.util.io as rmp_io

import tracemalloc

from memory_profiler import profile

@profile
def baseline_mem():
    # base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"
    # base_path = rmp_io.get_3dmatch_dir() + "/sun3d-home_at-home_at_scan1_2013_jan_1"

    # worldgen = FuseWorldGen(base_path)

    # Bundlefusion
    # worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    # worldgen.set_start(np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]))
    
    # Sun3d
    # worldgen.set_goal(np.array([ -4.0010950403757066, -1.1940135466560255, 0.66434728375427998 ]))
    # worldgen.set_start(np.array([ 4.9846375164171279, 1.3871051908909355, 0.51465610811519868 ]))
# 
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(120, seed=123)
    
    min_limits, max_limits = worldgen.get_world_limits()
    min_indices = np.array(min_limits) // worldgen.get_voxel_size()
    max_indices = np.array(max_limits) // worldgen.get_voxel_size()

    print(f"Min indices: {min_indices}")
    print(f"Max indices: {max_indices}")
    mem = np.prod(max_indices - min_indices) * 8 / 1024 / 1024
    print(f"Memory: {mem} MB")

    planner = PlannerFactory.baseline_planner_with_default_params()

    start = worldgen.get_start()
    goal = worldgen.get_goal()
    tsdf = worldgen.get_tsdf()

    planner.setup(start, goal, tsdf)
    
    planner.step()
    print(planner.success())

@profile
def learned_mem():
    # base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"
    base_path = rmp_io.get_3dmatch_dir() + "/sun3d-home_at-home_at_scan1_2013_jan_1"

    worldgen = FuseWorldGen(base_path)

    # Bundlefusion
    # worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    # worldgen.set_start(np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]))
    
    # Sun3d
    worldgen.set_goal(np.array([ -4.0010950403757066, -1.1940135466560255, 0.66434728375427998 ]))
    worldgen.set_start(np.array([ 4.9846375164171279, 1.3871051908909355, 0.51465610811519868 ]))

    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(120, seed=123)
    
    min_limits, max_limits = worldgen.get_world_limits()
    min_indices = np.array(min_limits) // worldgen.get_voxel_size()
    max_indices = np.array(max_limits) // worldgen.get_voxel_size()

    print(f"Min indices: {min_indices}")
    print(f"Max indices: {max_indices}")
    mem = np.prod(max_indices - min_indices) * 8 / 1024 / 1024
    print(f"Memory: {mem} MB")

    # model = ModelUtil.load_model("rckqzbvf", "latest")
    # model = ModelUtil.load_model("5msibfu3", "latest")
    model = ModelUtil.load_model("g2j8uxxd", "latest")
    model2 = copy.deepcopy(model)
    # model = ModelImporter.load_ffn() # From disk
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    raycasting_cuda_policy_params = RaycastingCudaPolicyParameters.from_yaml_general_config()
    learned_policy_params = LearnedPolicyRmpParameters.from_yaml_general_config()
    ray_observer_params = RayObserverParameters.from_yaml_general_config()
    planner_params = PlannerParameters.from_yaml_general_config()

    planner = PlannerFactory.learned_planner_minimal(model, raycasting_cuda_policy_params, learned_policy_params, ray_observer_params, planner_params)

    start = worldgen.get_start()
    goal = worldgen.get_goal()
    tsdf = worldgen.get_tsdf()

    planner.setup(start, goal, tsdf)
    
    planner.step()

    print(model2)


@profile
def rrt_mem():

    # base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"
    # base_path = rmp_io.get_3dmatch_dir() + "/sun3d-home_at-home_at_scan1_2013_jan_1"

    # worldgen = FuseWorldGen(base_path)

    # Bundlefusion
    # worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    # worldgen.set_start(np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]))
    
    # Sun3d
    # worldgen.set_goal(np.array([ -4.0010950403757066, -1.1940135466560255, 0.66434728375427998 ]))
    # worldgen.set_start(np.array([ 4.9846375164171279, 1.3871051908909355, 0.51465610811519868 ]))

    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(120, seed=123)

    min_limits, max_limits = worldgen.get_world_limits()
    min_indices = np.array(min_limits) // worldgen.get_voxel_size()
    max_indices = np.array(max_limits) // worldgen.get_voxel_size()

    print(f"Min indices: {min_indices}")
    print(f"Max indices: {max_indices}")
    mem = np.prod(max_indices - min_indices) * 8 / 1024 / 1024
    print(f"Memory: {mem} MB")

    planner = RRTPlanner(time=0.1, margin_to_obstacles=0.1) 
    esdf = worldgen.get_esdf()
    planner.set_esdf(esdf)
    planner.plan(worldgen.get_start(), worldgen.get_goal())

@profile
def chomp_mem():

    # base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"
    base_path = rmp_io.get_3dmatch_dir() + "/sun3d-home_at-home_at_scan1_2013_jan_1"

    worldgen = FuseWorldGen(base_path)

    # Bundlefusion
    # worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    # worldgen.set_start(np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]))
    
    # Sun3d
    worldgen.set_goal(np.array([ -4.0010950403757066, -1.1940135466560255, 0.66434728375427998 ]))
    worldgen.set_start(np.array([ 4.9846375164171279, 1.3871051908909355, 0.51465610811519868 ]))

    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(120, seed=123)

    min_limits, max_limits = worldgen.get_world_limits()
    min_indices = np.array(min_limits) // worldgen.get_voxel_size()
    max_indices = np.array(max_limits) // worldgen.get_voxel_size()

    print(f"Min indices: {min_indices}")
    print(f"Max indices: {max_indices}")
    mem = np.prod(max_indices - min_indices) * 8 / 1024 / 1024
    print(f"Memory: {mem} MB")

    planner = ChompPlanner(N=200)
    esdf = worldgen.get_esdf()
    planner.set_esdf(esdf)
    planner.plan(worldgen.get_start(), worldgen.get_goal())




if __name__ == "__main__":
    torch.use_deterministic_algorithms(False)

    baseline_mem()
    # learned_mem()
    # rrt_mem()

    # chomp_mem()

