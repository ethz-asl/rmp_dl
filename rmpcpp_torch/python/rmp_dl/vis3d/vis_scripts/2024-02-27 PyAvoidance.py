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
    worldgen = CustomWorldgenFactory.Overhang()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=3)


    model = ModelUtil.load_model("g2j8uxxd", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False, ray_vis=True, cpp_policy=False)
    
    model = ModelUtil.load_model("g2j8uxxd", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN - pypol",
                                            model=model, lstm_color=False, color=[1,0,0],
                                            step_vis=False, ray_vis=True, cpp_policy=True)
    
    planner3dvis.go()


def sun3d_home_at():
    base_path = rmp_io.get_3dmatch_dir() + "/sun3d-home_at-home_at_scan1_2013_jan_1"

    worldgen = FuseWorldGen(base_path)
    worldgen.set_goal(np.array([ -4.0010950403757066, -1.1940135466560255, 0.66434728375427998 ]))
    # worldgen.set_goal(np.array([ 4.9846375164171279, 1.3871051908909355, 0.51465610811519868 ]))
    
    planner3dvis = Planner3dVisComparison(worldgen, distancefield=False, draw_bbox=False, start=False)

    starts = [
        # np.array([ -4.0010950403757066, -1.1940135466560255, 0.66434728375427998 ]),
        np.array([ 4.9846375164171279, 1.3871051908909355, 0.51465610811519868 ]),
        np.array([ 4.1108104403361754, -1.5102357312875119, 0.74347017502066326 ]), 
        np.array([ -0.98599287383049505, -2.391314685699867, 0.39976777538079328 ]), 
        np.array([ -0.7865084015430962, 2.3160106207906792, 0.19492271209866313 ]), 
    ]

    for i, start in enumerate(starts):
        worldgen.set_start(start)

        # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.54, 0.94, 1.0], name=f"simple_target{i}")
        # PlannerChomp3dVisFactory.add_chomp_planner(planner3dvis, color=[1, 0.41, 0.7], name=f"chomp{i}")

        model = ModelUtil.load_model("g2j8uxxd", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(f"max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name=f"RNN{i}",
                                                model=model, lstm_color=True, color=[1,0,0],
                                                step_vis=False, ray_vis=True, cpp_policy=False)

        model = ModelUtil.load_model("g2j8uxxd", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(f"max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name=f"RNN-cpp{i}",
                                                model=model, lstm_color=False, color=[0,0,1],
                                                step_vis=False, ray_vis=True)

    planner3dvis.go()



if __name__ == "__main__":
    rnn_over_wall()
    # sun3d_home_at()

