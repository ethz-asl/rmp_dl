from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
import numpy as np
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.importer import ModelImporter
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import torch

import rmp_dl.util.io as rmp_io

def baseline_vs_ffn():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(190, seed=119000048)
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=340, initial_modifier=3)

    # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 0, 1])

    model = ModelUtil.load_model("5msibfu3", "latest")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN0",
                                            model=model, color=[1, 0, 0], 
                                            cpp_policy=False,
                                            step_vis=False, ray_vis=False)
    
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN10",
                                            model=model, color=[1, 1, 0], 
                                            step_vis=False, ray_vis=False, 
                                            cpp_policy=False,
                                            add_ray_noise=True, ray_noise_params={"noise_std": 0.1})
    
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN20",
                                            model=model, color=[0, 1, 0], 
                                            step_vis=False, ray_vis=False, 
                                            cpp_policy=False,
                                            add_ray_noise=True, ray_noise_params={"noise_std": 0.2})
    
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN30",
                                            model=model, color=[0, 1, 1], 
                                            step_vis=True, ray_vis=True, 
                                            cpp_policy=False,
                                            add_ray_noise=True, ray_noise_params={"noise_std": 0.3})
    
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN40",
                                            model=model, color=[0, 0, 1], 
                                            step_vis=False, ray_vis=False, 
                                            cpp_policy=False,
                                            add_ray_noise=True, ray_noise_params={"noise_std": 0.4})
    planner3dvis.go()


def ffn_stuck_wall():
    worldgen = CustomWorldgenFactory.Overhang()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=3)

    Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 0, 1])

    # model = ModelUtil.load_model("5msibfu3", "latest")
    model = ModelImporter.load_ffn() # From disk
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN",
                                            model=model, color=[1, 0, 0], 
                                            step_vis=False, ray_vis=True)
    
    planner3dvis.go()

def rnn_over_wall():
    worldgen = CustomWorldgenFactory.Overhang()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=3)


    # model = ModelUtil.load_model("g2j8uxxd", "last")
    model = ModelImporter.load_rnn() # From disk
    
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")

    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False, ray_vis=True)
    
    planner3dvis.go()


def rnn_stuck():
    worldgen = ProbabilisticWorldgenFactory.plane_world(35, seed=103500064)
    planner3dvis = Planner3dVis(worldgen, distancefield=True, initial_idx=1999, initial_modifier=3,
                                num_arrows=700, dx=(6.0, 0.5), dy=(6.0, 0.5), dz=(6.0, 0.12))


    model = ModelUtil.load_model("xivu80k3", "latest") # Older version of RNN
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False, ray_vis=True)

    planner3dvis.go()

def rnn_main_example_planes():
    worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=110000036)
    planner3dvis = Planner3dVis(worldgen, distancefield=False)

    model = ModelUtil.load_model("xivu80k3", "latest") # Older version of RNN
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False, ray_vis=True)
    
    planner3dvis.go()
        
def rnn_main_example_sphere_box():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
    planner3dvis = Planner3dVis(worldgen, distancefield=False)

    model = ModelUtil.load_model("xivu80k3", "latest") # Older version of RNN
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False, ray_vis=True)

    planner3dvis.go()

def bundle_apt2():
    base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"

    worldgen = FuseWorldGen(base_path)
    worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    
    planner3dvis = Planner3dVisComparison(worldgen, distancefield=False)

    starts = [
        np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]),
        np.array( [ -1.907390149036353, -0.12430902269179435, -0.23470042901280125 ]), 
        np.array([ 1.0270305715453296, -0.86594763482172399, -0.099932182750282872 ]), 
        np.array([ -3.6241529272109116, 0.82631784684181209, 2.1225966668831076 ]), 
        np.array([ -1.6582706697609855, -1.555483322901511, -0.69801436974119879 ]), 
    ]

    for i, start in enumerate(starts):
        worldgen.set_start(start)

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.54, 0.94, 1.0], name=f"simple_target{i}")
        # PlannerChomp3dVisFactory.add_chomp_planner(planner3dvis, color=[0, 0, 1], name=f"chomp{i}")

        model = ModelUtil.load_model("g2j8uxxd", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(f"max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name=f"RNN{i}",
                                                model=model, lstm_color=True, color=[1,0,0],
                                                step_vis=False, ray_vis=True)


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

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.54, 0.94, 1.0], name=f"simple_target{i}")
        PlannerComparison3dVisFactory.add_chomp_planner(planner3dvis, color=[1, 0.41, 0.7], name=f"chomp{i}")

        model = ModelUtil.load_model("g2j8uxxd", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(f"max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name=f"RNN{i}",
                                                model=model, lstm_color=True, color=[1,0,0],
                                                step_vis=False, ray_vis=True)

    planner3dvis.go()

def bundle_apt0():
    base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt0"

    worldgen = FuseWorldGen(base_path)
    worldgen.set_goal(np.array([ -0.27697324648179789, 0.59887816677365935, 1.7690977367486396 ]))
    planner3dvis = Planner3dVisComparison(worldgen, distancefield=False)
    
    starts = [
        np.array([ -4.0682302171091784, 1.0622816854589434, 1.8616656097476989 ]),
        np.array([ -3.1565721179339814, -1.6870687642286855, -2.0969505820044305 ]),
        np.array([ -2.907117469991054, -1.7286157637288198, -0.081385690890984769 ]),
        np.array([ -0.62506188899845139, -2.7286688533322958, -1.0476755293992335 ]),

    ]

    for i, start in enumerate(starts):
        worldgen.set_start(start)

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.54, 0.94, 1.0], name=f"simple_target{i}")
        # PlannerChomp3dVisFactory.add_chomp_planner(planner3dvis, color=[1, 0.41, 0.7], name=f"chomp{i}")

        model = ModelUtil.load_model("g2j8uxxd", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(f"max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name=f"RNN{i}",
                                                model=model, lstm_color=True, color=[1,0,0],
                                                step_vis=False, ray_vis=True)


    planner3dvis.go()


def bundle_apt2_baseline():
    base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"

    worldgen = FuseWorldGen(base_path)
    worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    
    planner3dvis = Planner3dVisComparison(worldgen, distancefield=False)

    starts = [
        np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]),
        np.array([ -3.6241529272109116, 0.82631784684181209, 2.1225966668831076 ]), 
    ]
    colors = [
        [1, 0, 0], 
        [0, 0, 1]
    ]

    for i, start in enumerate(starts):
        worldgen.set_start(start)

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=colors[i], name=f"simple_target{i}")

    planner3dvis.go()

if __name__ == "__main__":
    torch.use_deterministic_algorithms(False)
    baseline_vs_ffn()
    # ffn_stuck_wall()
    # rnn_over_wall()
    # rnn_stuck()
    # rnn_main_example_planes()
    # rnn_main_example_sphere_box()
    # bundle_apt2()
    # sun3d_home_at()
    # bundle_apt0()
    # bundle_apt2_baseline()

