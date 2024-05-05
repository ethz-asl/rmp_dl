import os
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
import numpy as np
from rmp_dl.learning.data.pipeline.nodes.worlds.probabilistic_world import ProbabilisticWorld
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.exporter import ModelExporter
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import torch

import rmp_dl.util.io as rmp_io


def msw():
    # worldgen = CustomWorldgenFactory.SlitWall()
    worldgen = CustomWorldgenFactory.MultipleSlitWall()
    # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
    planner3dvis = Planner3dVis(worldgen, distancefield=True, num_arrows=8e3, distancefield_inflation=0.8)
    
    # Planner3dVisFactory.add_expert_planner(planner3dvis, worldgen)

    planner3dvis.go()

def test():
    # worldgen = CustomWorldgenFactory.SlitWall()
    # worldgen = CustomWorldgenFactory.MultipleSlitWall()
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=1)
    planner3dvis = Planner3dVis(worldgen, distancefield=True, num_arrows=8e3, distancefield_inflation=0.3, 
                                dx=(5.0, 1.0), dy=(5.0, 1.0), dz=(5.0, 1.0))
    
    # Planner3dVisFactory.add_expert_planner(planner3dvis, worldgen)

    planner3dvis.go()

def test2():
    # worldgen = CustomWorldgenFactory.SlitWall()
    # worldgen = CustomWorldgenFactory.MultipleSlitWall()
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=21)
    print(worldgen.get_distancefield(inflation=0.25).fraction_valid_and_reachable())
    planner3dvis = Planner3dVis(worldgen, distancefield=True, num_arrows=8e3, distancefield_inflation=0.2)
    
    # Planner3dVisFactory.add_expert_planner(planner3dvis, worldgen)

    planner3dvis.go()

def reachability_test():
    for obstacles in [100, 200]:
        for inflation in [0.1, 0.2, 0.3, 0.4]:
            for i in range(3):
                worldgen = ProbabilisticWorldgenFactory.sphere_box_world(obstacles, seed=i)
                df: DistanceFieldGen = worldgen.get_distancefield(inflation=inflation)
                valid, reachable  = df.fraction_valid_and_reachable()
                print(f"Inflation: {inflation}, Seed: {i}, Fraction valid: {valid:.2f}, Fraction reachable: {reachable:.2f}")

def plane_tests():
    # worldgen = CustomWorldgenFactory.SlitWall()
    # worldgen = CustomWorldgenFactory.MultipleSlitWall()
    worldgen = ProbabilisticWorldgenFactory.plane_world(85, seed=308500010)

    worldgen = ProbabilisticWorld(
        name="hoi",
        experiment_type="validation",
        seed=10,
        start_goal_location_type = "random_with_mindist",
        start_goal_location_type_params = {"min_dist": 7.0},
        start_goal_margin_to_obstacles=0.5, 
        world_limits=[[-0.2, -0.2, -0.2], [10.2, 10.2, 10.2]],
        voxel_truncation_distance_vox=4,
        voxel_size=0.2,
        obstacle_groups=[
            {
                "name": "planes",
                "group_type": "weighted",
                "count": 85,
                "obstacles": [
                    {
                        "obstacle_type": "Box",
                        "length_x": 0.3,
                        "length_y": "NORMAL(3.0, 2.0)",
                        "length_z": "NORMAL(3.0, 2.0)"
                    },
                    {
                        "obstacle_type": "Box",
                        "length_x": "NORMAL(3.0, 2.0)",
                        "length_y": 0.3,
                        "length_z": "NORMAL(3.0, 2.0)"
                    },
                    {
                        "obstacle_type": "Box",
                        "length_x": "NORMAL(3.0, 2.0)",
                        "length_y": "NORMAL(3.0, 2.0)",
                        "length_z": 0.3
                    },
                ],
            }
        ]
    ).get_world_constructor()()

    planner3dvis = Planner3dVis(worldgen, distancefield=True, num_arrows=8e3, distancefield_inflation=0.0, )
                                # dx=(5.0, 1.0), dy=(5.0, 1.0), dz=(5.0, 1.0))
    
    # Planner3dVisFactory.add_expert_planner(planner3dvis, worldgen)

    planner3dvis.go()

        

if __name__ == "__main__":
    msw()
    # test()
    # reachability_test()
    # plane_tests()


