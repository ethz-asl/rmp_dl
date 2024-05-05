from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
import numpy as np
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.importer import ModelImporter
from rmp_dl.vis3d.cinema.camera import CameraBase
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import torch

import rmp_dl.util.io as rmp_io

def ffn_wall():
    worldgen = CustomWorldgenFactory.SingleWallEmptyLargeWorld()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=3)

    # model = ModelUtil.load_model("5msibfu3", "latest")
    model = ModelImporter.load_ffn() # From disk
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name="FFN",
                                            model=model, color=[1, 0, 0], 
                                            step_vis=False, ray_vis=True, 
                                            use_meshes=True, mesh_radius=0.1, ray_pred_size=2.0)
    
    planner3dvis.go()


def rotating_example_sb():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=1293)
    planner3dvis = Planner3dVis(worldgen, distancefield=False)

    model = ModelUtil.load_model("g2j8uxxd", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False, ray_vis=True)
    
    planner3dvis.go()
        
def rotating_example_planes():
    # worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=834)
    worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=8343)
    planner3dvis = Planner3dVis(worldgen, distancefield=False)

    model = ModelUtil.load_model("g2j8uxxd", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False, ray_vis=True)

    planner3dvis.go()


def ffn_stuck_wall():
    worldgen = CustomWorldgenFactory.Overhang()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=1)

    # model = ModelUtil.load_model("5msibfu3", "latest")
    model = ModelImporter.load_ffn() # From disk
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN",
                                            model=model, color=[0, 0, 1], 
                                            step_vis=False,
                                            ray_vis=True, ray_pred_size=1.0,
                                            use_meshes=True, mesh_radius=0.03, 
                                            )
    

    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ -0.91969816980532559, 0.21107579712240565, -0.33106235715628479 ],
			"lookat" : [ 5.5320956376916293, 6.3138969835865204, 4.8524991570063891 ],
			"up" : [ 0.19789008182448128, 0.97746754870821262, 0.073462281055369991 ],
			"zoom" : 0.22245247515740024
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()
    
    img_dir = rmp_io.resolve_directory("figures/rss_pres/ffn_stuck_wall.png")
    planner3dvis.plot3d.vis.capture_screen_image(img_dir, True)

    planner3dvis.go()

def rnn_over_wall():
    worldgen = CustomWorldgenFactory.Overhang()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=1)

    # model = ModelUtil.load_model("5msibfu3", "latest")
    model = ModelImporter.load_ffn() # From disk
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN",
                                            model=model, color=[0, 0, 1], 
                                            step_vis=False,
                                            )

    model = ModelImporter.load_rnn() # From disk
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, color=[1, 0, 1], 
                                            step_vis=False,
                                            ray_vis=True, ray_pred_size=1.0,
                                            use_meshes=True, mesh_radius=0.03, 
                                            )
    

    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ -0.91969816980532559, 0.21107579712240565, -0.33106235715628479 ],
			"lookat" : [ 5.5320956376916293, 6.3138969835865204, 4.8524991570063891 ],
			"up" : [ 0.19789008182448128, 0.97746754870821262, 0.073462281055369991 ],
			"zoom" : 0.22245247515740024
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()
    
    img_dir = rmp_io.resolve_directory("figures/rss_pres/rnn_over_wall.png")
    planner3dvis.plot3d.vis.capture_screen_image(img_dir, True)

    planner3dvis.go()

def teaser_2stage1():
    base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"

    worldgen = FuseWorldGen(base_path)
    worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    
    planner3dvis = Planner3dVis(worldgen, distancefield=False, draw_bbox=False, start=False)

    starts = [
        np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]),
        np.array( [ -1.907390149036353, -0.12430902269179435, -0.23470042901280125 ]), 
        np.array([ 1.0270305715453296, -0.86594763482172399, -0.099932182750282872 ]), 
        np.array([ -3.6241529272109116, 0.82631784684181209, 2.1225966668831076 ]), 
        np.array([ -1.6582706697609855, -1.555483322901511, -0.69801436974119879 ]), 
    ]


    for i, start in enumerate(starts):
        worldgen.set_start(start)

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.129, 0.46, 1.0], name=f"simple_target{i}", trajectory_size=0.03)

        # model = ModelUtil.load_model("g2j8uxxd", "last")
        # model = RayModelDirectionConversionWrapper(model)
        # model.set_output_decoder_from_factory(f"max_sum50_decoder")
        # Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name=f"RNN{i}",
        #                                         model=model, lstm_color=True, color=[1,0,0],
        # #                                         step_vis=False, ray_vis=True, trajectory_size=0.03)
        

    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
        "pos" : [ -0.52838845436118331, 0.44258398944489724, -0.72451711752369174 ],
        "lookat" : [ -0.81990250574760948, 1.3197161195871054, 2.2511695921109687 ],
        "up" : [ 0.7882283327331534, 0.57279388432616352, -0.22495168715806288 ],
        "zoom" : 0.53999999999999981
        }
    )



    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()

    img_dir = rmp_io.resolve_directory("figures/rss_pres/teaser_2stage1.png")
    planner3dvis.plot3d.vis.capture_screen_image(img_dir, True)
    planner3dvis.go()

def teaser_2stage2():
    base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"

    worldgen = FuseWorldGen(base_path)
    worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    
    planner3dvis = Planner3dVis(worldgen, distancefield=False, draw_bbox=False, start=False)

    starts = [
        np.array([ -3.6780462076087486, 3.3998243200873044, 2.629309194238735 ]),
        np.array( [ -1.907390149036353, -0.12430902269179435, -0.23470042901280125 ]), 
        np.array([ 1.0270305715453296, -0.86594763482172399, -0.099932182750282872 ]), 
        np.array([ -3.6241529272109116, 0.82631784684181209, 2.1225966668831076 ]), 
        np.array([ -1.6582706697609855, -1.555483322901511, -0.69801436974119879 ]), 
    ]


    for i, start in enumerate(starts):
        worldgen.set_start(start)

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.129, 0.46, 1.0], name=f"simple_target{i}", trajectory_size=0.03)

        model = ModelUtil.load_model("g2j8uxxd", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory(f"max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner_minimal_with_ray_observer(planner3dvis, name=f"RNN{i}",
                                                model=model, lstm_color=True, color=[1,0,0],
                                                step_vis=False, ray_vis=True, trajectory_size=0.03)
        

    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
        "pos" : [ -0.52838845436118331, 0.44258398944489724, -0.72451711752369174 ],
        "lookat" : [ -0.81990250574760948, 1.3197161195871054, 2.2511695921109687 ],
        "up" : [ 0.7882283327331534, 0.57279388432616352, -0.22495168715806288 ],
        "zoom" : 0.53999999999999981
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()

    img_dir = rmp_io.resolve_directory("figures/rss_pres/teaser_2stage2.png")
    planner3dvis.plot3d.vis.capture_screen_image(img_dir, True)
    planner3dvis.go()

def camera_view_ray_gif():
    worldgen = CustomWorldgenFactory.SimpleWorld()

    planner3dvis = Planner3dVis(worldgen)
    planner3dvis.go()

if __name__ == "__main__":
    torch.use_deterministic_algorithms(False)
    # ffn_wall()
    # rotating_example_sb()
    # rotating_example_planes()

    # ffn_stuck_wall()
    # rnn_over_wall()
    # teaser_2stage2()

    camera_view_ray_gif()

