import os
from comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
import numpy as np
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.vis3d.cinema.camera import CameraBase
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.planner3dvis_factory import Planner3dVisFactory
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
import torch

import rmp_dl.util.io as rmp_io
    
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt

def save_colorbar(file_name):
    # Use the get_RGB function to get the corresponding colors
    cvalues = np.linspace(0, 1, 10000) 
    colors = Open3dUtils.get_RGB(np.array(cvalues), minnorm=0)

    # Create a colormap from the generated colors
    cmap = mcolors.ListedColormap(colors[int(len(colors) / 3.0):])

    # Create and display the colorbar
    fig, ax = plt.subplots(figsize=(2.2, 1.8))
    cb = mcolorbar.ColorbarBase(ax, cmap=cmap, orientation='vertical')

    plt.rcParams['text.usetex'] = True
    
    cb.set_ticks([0, 1])  # Setting ticks at the ends of the colorbar
    cb.set_ticklabels([f'Low LSTM Influence', f'High LSTM Influence'])  # Labeling the ticks

    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    cb.ax.set_ylim([0, 1])

    fig.tight_layout()
    fig.savefig(file_name, dpi=400, transparent=True)

def baseline_vs_ffn():
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(190, seed=119000048)
    # worldgen = ProbabilisticWorldgenFactory.plane_world(75, seed=107500058)
    # worldgen = ProbabilisticWorldgenFactory.world25d(20, seed=0)
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=340, initial_modifier=3)

    Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.129, 0.46, 1.0], trajectory_size=.01)

    model = ModelUtil.load_model("5msibfu3", "latest")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN",
                                            model=model, color=[1, 0, 0], 
                                            step_vis=True, step_vis_choices={"goal"},
                                            ray_vis=True, ray_pred_size=0.5,
                                            use_meshes=True, mesh_radius=0.006, 
                                            trajectory_size=0.01)
    
    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ -0.95722462981361611, 0.28566976867730054, -0.045975986580424906 ],
			"lookat" : [ 5.0790628902180073, 5.8213552399162207, 4.3321645637346631 ],
			"up" : [ 0.28667320071171981, 0.95788087180941306, -0.016814024364701759 ],
			"zoom" : 0.18824311624928119
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()
    
    img_dir = rmp_io.resolve_directory("figures/report3/ffn_vs_baseline.png")

    planner3dvis.plot3d.vis.capture_screen_image(img_dir, True)
    # planner3dvis.go()
    planner3dvis.destroy()


def ffn_stuck_wall():
    worldgen = CustomWorldgenFactory.Overhang()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=1)

    Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.129, 0.46, 1.0], trajectory_size=.05)

    model = ModelUtil.load_model("5msibfu3", "latest")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN",
                                            model=model, color=[1, 0, 0], 
                                            step_vis=False,
                                            ray_vis=True, ray_pred_size=1.2,
                                            use_meshes=True, mesh_radius=0.06, 
                                            trajectory_size=0.05)
    
    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ -0.92918243376359855, 0.34301851070390654, -0.13768916478659476 ],
			"lookat" : [ 5.316396175894333, 6.4882976596235187, 4.7136014989188508 ],
			"up" : [ 0.3376931111764534, 0.93927268060833136, 0.061075315200228748 ],
			"zoom" : 0.34824311624928123
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()
    
    img_dir = rmp_io.resolve_directory("figures/report3/ffn_stuck_wall.png")

    planner3dvis.plot3d.vis.capture_screen_image(img_dir, True)
    # planner3dvis.go()
    planner3dvis.destroy()

def rnn_over_wall():
    worldgen = CustomWorldgenFactory.Overhang()
    planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=3)

    # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 0, 1])

    model = ModelUtil.load_model("xivu80k3", "last")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False,
                                            ray_vis=True, ray_pred_size=1.2,
                                            use_meshes=True, mesh_radius=0.06, 
                                            trajectory_size=0.05)
    
    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ -0.92918243376359855, 0.34301851070390654, -0.13768916478659476 ],
			"lookat" : [ 5.316396175894333, 6.4882976596235187, 4.7136014989188508 ],
			"up" : [ 0.3376931111764534, 0.93927268060833136, 0.061075315200228748 ],
			"zoom" : 0.34824311624928123
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()

    img_dir = rmp_io.resolve_directory("figures/report3/rnn_over_wall")

    # We do a sum, but there is only 1 value. If there were multiple LSTMs, summing would be slightly strange. 
    # But with 1 LSTM it works fine
    diffs = [sum(obs["learned_policy"]["recurrent_diff_norm"].values()) for obs in planner3dvis.observations["RNN"]]
    planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/image.png", True)

    save_colorbar(img_dir + "/colorbar.png")

    # Load the existing image and the colorbar
    o3d_img = Image.open(img_dir + "/image.png")
    colorbar_img = Image.open(img_dir + "/colorbar.png")

    colorbar_img = colorbar_img.resize((colorbar_img.width // 2, colorbar_img.height // 2))

    # Create a new image to accommodate both
    new_img = Image.new('RGB', (o3d_img.width, o3d_img.height))
    new_img.paste(o3d_img, (0, 0))
    new_img.paste(colorbar_img, (400, 100), colorbar_img)

    # Save or display the resulting image
    new_img.save(img_dir + "/rnn_over_wall.png")

    # planner3dvis.go()
    planner3dvis.destroy()


def rnn_stuck():
    worldgen = ProbabilisticWorldgenFactory.plane_world(35, seed=103500064)
    planner3dvis = Planner3dVis(worldgen, distancefield=True, initial_idx=1999, initial_modifier=3,
                                num_arrows=700, dx=(6.0, 0.5), dy=(6.0, 0.5), dz=(6.0, 0.12))

    # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 0, 1])

    # Using latest instead of last means that we have slightly less trained model. However the vizualization looks
    # much better for the older one. For this visualization we don't care much about what exact model, as 
    # I'm just using it to explain the multi-modality. So I use a slightly worse model here. 
    model = ModelUtil.load_model("xivu80k3", "latest")
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")
    Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                            model=model, lstm_color=True,
                                            step_vis=False,
                                            ray_vis=True, ray_pred_size=1.6,
                                            use_meshes=True, mesh_radius=0.05)
    
    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ 0.81428735772882144, -0.53886376649254142, 0.21578215914316537 ],
			"lookat" : [ 5.9872465111871049, 9.0751053302987614, 8.3432958467072869 ],
			"up" : [ -0.16874293325685785, 0.13593468165173392, 0.97624156068061019 ],
			"zoom" : 0.39823470742191502
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()
    
    img_dir = rmp_io.resolve_directory("figures/report3/rnn_stuck.png")

    planner3dvis.plot3d.vis.capture_screen_image(img_dir, True)
    # planner3dvis.go()
    planner3dvis.destroy()

def rnn_main_example_planes(images_only=False):

    img_dir = rmp_io.resolve_directory("figures/report3/rnn_main_example_planes")
    if not images_only:
        worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=110000036)
        # worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=110000054)
        planner3dvis = Planner3dVis(worldgen, distancefield=False)

        # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 0, 1])

        model = ModelUtil.load_model("xivu80k3", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory("max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                                model=model, lstm_color=True,
                                                step_vis=False,
                                                ray_vis=True, ray_pred_size=1.2,
                                                use_meshes=False, mesh_radius=0.017,
                                                trajectory_size=0.03)
        
        # We also do a 2nd view, that zooms in on the local minimum
        # We do this view first, as we add a 2nd planning run with a thicker trajectory afterwards
        view = CameraBase.View(
            **{
                # This is here for reproducibility. To get initial nice view parameters simply run this script, 
                # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
                # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ 0.53047378287888569, 0.19261881987924731, -0.82552744103784392 ],
			"lookat" : [ 4.6093049069423211, 4.0863510665159861, 6.0061065410341365 ],
			"up" : [ -0.081889475557321081, 0.98093184006578771, 0.17625787624414069 ],
			"zoom" : 0.18824311624928114
            }
        )


        vc = planner3dvis.plot3d.vis.get_view_control()
        vc.set_lookat(view.lookat)
        vc.set_front(view.pos)
        vc.set_up(view.up)
        vc.set_zoom(view.zoom)
        planner3dvis.plot3d.vis.poll_events()
        planner3dvis.plot3d.vis.update_renderer()

        planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/image-zoomed.png", True)
        
        # We do an identical run, but with a thicker trajectory such that the zoomed out screenshot looks nicer
        Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN_THICK",
                                                model=model, lstm_color=True,
                                                step_vis=False,
                                                ray_vis=True, ray_pred_size=1.2,
                                                use_meshes=False, mesh_radius=0.017,
                                                trajectory_size=0.1)
        
        view = CameraBase.View(
            **{
                # This is here for reproducibility. To get initial nice view parameters simply run this script, 
                # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
                # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
			"pos" : [ 0.20024690612975268, -0.82973705124552333, 0.5209967393140249 ],
			"lookat" : [ 5.0504945408150439, 4.0068900083193579, 5.6271290367290767 ],
			"up" : [ 0.97963005606505538, 0.16140538284107692, -0.1194707313273928 ],
			"zoom" : 0.96824311624928172
            }
        )


        vc = planner3dvis.plot3d.vis.get_view_control()
        vc.set_lookat(view.lookat)
        vc.set_front(view.pos)
        vc.set_up(view.up)
        vc.set_zoom(view.zoom)
        planner3dvis.plot3d.vis.poll_events()
        planner3dvis.plot3d.vis.update_renderer()
        planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/image.png", True)
        
        # We do a sum, but there is only 1 value. If there were multiple LSTMs, summing would be slightly strange. 
        # But with 1 LSTM it works fine
        diffs = [sum(obs["learned_policy"]["recurrent_diff_norm"].values()) for obs in planner3dvis.observations["RNN"]]

        save_colorbar(img_dir + "/colorbar.png")
        # planner3dvis.go()
        planner3dvis.destroy()

    # Load the existing image and the colorbar
    o3d_img = Image.open(img_dir + "/image.png")
    colorbar_img = Image.open(img_dir + "/colorbar.png")

    colorbar_img = colorbar_img.resize((int(colorbar_img.width / 3), int(colorbar_img.height / 3)))

    # Create a new image to accommodate both
    new_img = Image.new('RGB', (o3d_img.width, o3d_img.height))
    new_img.paste(o3d_img, (0, 0))
    new_img.paste(colorbar_img, (1150, 262), colorbar_img)

    # Load the zoomed in image
    zoomed_img = Image.open(img_dir + '/image-zoomed.png')

    # Define the rectangle coordinates on the original image
    # For example, (left, upper, right, lower) 
    rect_coords = (934, 404, 1077, 586)  

    # Draw a red rectangle on the original image
    draw = ImageDraw.Draw(new_img)
    draw.rectangle(rect_coords, outline="red", width=3)

    # Crop the zoomed image 
    crop_coords = (164, 228, 1382, 906) 
    zoomed_img = zoomed_img.crop(crop_coords)
    zoomed_img = zoomed_img.resize(size=(int(zoomed_img.width / 3.5), int(zoomed_img.height / 3.5)))

    # Add a red border around the zoomed image
    border_width = 10  # Adjust border width as needed
    bordered_zoom = Image.new('RGB', 
                            (zoomed_img.width + 2 * border_width, zoomed_img.height + 2 * border_width), 
                            'red')
    bordered_zoom.paste(zoomed_img, (border_width, border_width))

    # Place the zoomed image on the original image
    # Example: top-right corner
    place_position = (650, 650)  # Adjust this as needed
    new_img.paste(bordered_zoom, place_position)

    crop_coords = (600, 170, 1402, 940) 
    new_img = new_img.crop(crop_coords)

    # Save or display the resulting image
    new_img.save(img_dir + "/rnn_main_example_planes.png")



    # planner3dvis.go()
def rnn_main_example_sphere_box(images_only=False):
    img_dir = rmp_io.resolve_directory("figures/report3/rnn_main_example_sphere_box")
    def plot_and_save():
        worldgen = ProbabilisticWorldgenFactory.sphere_box_world(200, seed=120000064)
        planner3dvis = Planner3dVis(worldgen, distancefield=False)

        # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0, 0, 1])

        model = ModelUtil.load_model("xivu80k3", "last")
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory("max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                                model=model, lstm_color=True,
                                                step_vis=False,
                                                ray_vis=True, ray_pred_size=1.2,
                                                use_meshes=False, mesh_radius=0.017, 
                                                trajectory_size=0.03)
        
        # We also do a 2nd view, that zooms in on the local minimum
        # We do this view first, as we add a 2nd planning run with a thicker trajectory afterwards
        view = CameraBase.View(
            **{
                # This is here for reproducibility. To get initial nice view parameters simply run this script, 
                # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
                # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
                "pos" : [ 0.37694810781844962, -0.053270835327632313, -0.92470121775392233 ],
                "lookat" : [ 7.2124910730740455, 6.3930839033957163, 5.5748569981518949 ],
                "up" : [ -0.013003151740587835, 0.99794199986166687, -0.062790787197709835 ],
                "zoom" : 0.19939638427020448
            }
        )

        vc = planner3dvis.plot3d.vis.get_view_control()
        vc.set_lookat(view.lookat)
        vc.set_front(view.pos)
        vc.set_up(view.up)
        vc.set_zoom(view.zoom)
        planner3dvis.plot3d.vis.poll_events()
        planner3dvis.plot3d.vis.update_renderer()

        planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/image-zoomed.png", True)

        Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN_THICK",
                                                model=model, lstm_color=True,
                                                step_vis=False,
                                                ray_vis=True, ray_pred_size=1.2,
                                                use_meshes=False, mesh_radius=0.017, 
                                                trajectory_size=0.1)

        # 1st zoomed out view
        # We do an identical run, but with a thicker trajectory such that the zoomed out screenshot looks nicer
        view = CameraBase.View(
            **{
                # This is here for reproducibility. To get initial nice view parameters simply run this script, 
                # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
                # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
                "pos" : [ 0.92634279406669562, -0.12833552926993594, -0.35414547831041576 ],
                "lookat" : [ 6.1267831808618505, 4.9340481507051601, 4.2291530999054556 ],
                "up" : [ 0.11709237416037817, 0.99170079768896924, -0.053093349645206699 ],
                "zoom" : 0.96874102037947696
            }
        )

        vc = planner3dvis.plot3d.vis.get_view_control()
        vc.set_lookat(view.lookat)
        vc.set_front(view.pos)
        vc.set_up(view.up)
        vc.set_zoom(view.zoom)
        planner3dvis.plot3d.vis.poll_events()
        planner3dvis.plot3d.vis.update_renderer()

        planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/image.png", True)

        # We do a sum, but there is only 1 value. If there were multiple LSTMs, summing would be slightly strange. 
        # But with 1 LSTM it works fine
        diffs = [sum(obs["learned_policy"]["recurrent_diff_norm"].values()) for obs in planner3dvis.observations["RNN"]]
        save_colorbar(img_dir + "/colorbar.png")
        # planner3dvis.go()
        planner3dvis.destroy()

    if not images_only:
        plot_and_save()

    # Load the existing image and the colorbar
    o3d_img = Image.open(img_dir + "/image.png")
    colorbar_img = Image.open(img_dir + "/colorbar.png")

    colorbar_img = colorbar_img.resize((int(colorbar_img.width / 3), int(colorbar_img.height / 3)))

    # Create a new image to accommodate both
    new_img = Image.new('RGB', (o3d_img.width, o3d_img.height))
    new_img.paste(o3d_img, (0, 0))
    new_img.paste(colorbar_img, (960, 180), colorbar_img)

    # Load the zoomed in image
    zoomed_img = Image.open(img_dir + '/image-zoomed.png')

    # Define the rectangle coordinates on the original image
    # For example, (left, upper, right, lower) 
    rect_coords = (832, 353, 957, 481)  

    # Draw a red rectangle on the original image
    draw = ImageDraw.Draw(new_img)
    draw.rectangle(rect_coords, outline="red", width=3)

    # Crop the zoomed image 
    crop_coords = (335, 34, 1390, 789) 
    zoomed_img = zoomed_img.crop(crop_coords)
    zoomed_img = zoomed_img.resize(size=(int(zoomed_img.width / 3), int(zoomed_img.height / 3)))

    # Add a red border around the zoomed image
    border_width = 10  # Adjust border width as needed
    bordered_zoom = Image.new('RGB', 
                            (zoomed_img.width + 2 * border_width, zoomed_img.height + 2 * border_width), 
                            'red')
    bordered_zoom.paste(zoomed_img, (border_width, border_width))

    # Place the zoomed image on the original image
    # Example: top-right corner
    place_position = (602, 514)  # Adjust this as needed
    new_img.paste(bordered_zoom, place_position)

    crop_coords = (510, 88, 1245, 863) 
    new_img = new_img.crop(crop_coords)

    # Save or display the resulting image
    new_img.save(img_dir + "/rnn_main_example_sphere_box.png")



def bundle_apt2(images_only=False):
    img_dir = rmp_io.resolve_directory("figures/report3/example_bundle_apt2")

    def plot_and_save():
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
        planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/image.png", True)

    if not images_only:
        plot_and_save()
    
    # We just do a generic colorbar
    save_colorbar(img_dir + "/colorbar.png")

    # Load the existing image and the colorbar
    o3d_img = Image.open(img_dir + "/image.png")
    colorbar_img = Image.open(img_dir + "/colorbar.png")

    colorbar_img = colorbar_img.resize((int(colorbar_img.width / 2.5), int(colorbar_img.height / 2.5)))

    # Create a new image to accommodate both
    new_img = Image.new('RGB', (o3d_img.width, o3d_img.height))
    new_img.paste(o3d_img, (0, 0))
    new_img.paste(colorbar_img, (1276, 7), colorbar_img)


    new_img.save(img_dir + "/example_bundle_apt2.png")

    


def sun3d_home_at(images_only=False, chomp=False):
    img_dir = rmp_io.resolve_directory("figures/report3/example_sun3d_home_at")

    def plot_and_save():
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

            Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.129, 0.46, 1.0], name=f"simple_target{i}", trajectory_size=0.03)
            if chomp:
                PlannerComparison3dVisFactory.add_chomp_planner(planner3dvis, color=[1, 0.41, 0.7], name=f"chomp{i}")

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
			"pos" : [ 0.22749611502927317, 0.52489862625556583, 0.82019933540670931 ],
			"lookat" : [ 0.22751237860911286, -0.38374976237127506, -1.017566380236486 ],
			"up" : [ -0.13270182034430864, -0.81771613351321626, 0.56011655204029132 ],
			"zoom" : 0.2999999999999996
            }
        )


        vc = planner3dvis.plot3d.vis.get_view_control()
        vc.set_lookat(view.lookat)
        vc.set_front(view.pos)
        vc.set_up(view.up)
        vc.set_zoom(view.zoom)
        planner3dvis.plot3d.vis.poll_events()
        planner3dvis.plot3d.vis.update_renderer()
        img_name = "/image.png" if not chomp else "/image_chomp.png"
        planner3dvis.plot3d.vis.capture_screen_image(img_dir + img_name, True)

    if not images_only:
        plot_and_save()
    
    # We just do a generic colorbar
    save_colorbar(img_dir + "/colorbar.png")

    # Load the existing image and the colorbar
    img_name = "/image.png" if not chomp else "/image_chomp.png"
    o3d_img = Image.open(img_dir + img_name)
    colorbar_img = Image.open(img_dir + "/colorbar.png")

    colorbar_img = colorbar_img.resize((int(colorbar_img.width / 2.5), int(colorbar_img.height / 2.5)))

    # Create a new image to accommodate both
    new_img = Image.new('RGB', (o3d_img.width, o3d_img.height))
    new_img.paste(o3d_img, (0, 0))
    # new_img.paste(colorbar_img, (1469, 400), colorbar_img)

    img_name = "/example_sun3d_home_at.png" if not chomp else "/example_sun3d_home_at_chomp.png"
    new_img.save(img_dir + img_name)

    # planner3dvis.go()



def bundle_apt0(images_only=False):
    img_dir = rmp_io.resolve_directory("figures/report3/example_bundle_apt0")

    def plot_and_save():
        base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt0"

        worldgen = FuseWorldGen(base_path)
        worldgen.set_goal(np.array([ -0.27697324648179789, 0.59887816677365935, 1.7690977367486396 ]))
        planner3dvis = Planner3dVis(worldgen, distancefield=False, start=False, draw_bbox=False)
        
        starts = [
            np.array([ -4.0682302171091784, 1.0622816854589434, 1.8616656097476989 ]),
            np.array([ -3.1565721179339814, -1.6870687642286855, -2.0969505820044305 ]),
            np.array([ -2.907117469991054, -1.7286157637288198, -0.081385690890984769 ]),
            np.array([ -0.62506188899845139, -2.7286688533322958, -1.0476755293992335 ]),
            np.array([ -3.1916431707249093, -0.67258039125373392, -0.24110959375175431 ])
        ]

        for i, start in enumerate(starts):
            worldgen.set_start(start)

            # Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[0.129, 0.46, 1.0], name=f"simple_target{i}", trajectory_size=0.03)

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
			"pos" : [ -0.78841127217438167, 0.30062296978470976, -0.53668752169786349 ],
			"lookat" : [ -1.4241874643756505, -0.56364711725137562, 0.30083928695040069 ],
			"up" : [ 0.57972135030919103, 0.65490989132900002, -0.48478468440651634 ],
			"zoom" : 0.57999999999999985
            }
        )


        vc = planner3dvis.plot3d.vis.get_view_control()
        vc.set_lookat(view.lookat)
        vc.set_front(view.pos)
        vc.set_up(view.up)
        vc.set_zoom(view.zoom)
        planner3dvis.plot3d.vis.poll_events()
        planner3dvis.plot3d.vis.update_renderer()
        planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/image.png", True)

    if not images_only:
        plot_and_save()
    
    # We just do a generic colorbar
    save_colorbar(img_dir + "/colorbar.png")

    # Load the existing image and the colorbar
    o3d_img = Image.open(img_dir + "/image.png")
    colorbar_img = Image.open(img_dir + "/colorbar.png")

    colorbar_img = colorbar_img.resize((int(colorbar_img.width / 2.5), int(colorbar_img.height / 2.5)))

    # Create a new image to accommodate both
    new_img = Image.new('RGB', (o3d_img.width, o3d_img.height))
    new_img.paste(o3d_img, (0, 0))
    new_img.paste(colorbar_img, (1428, 62), colorbar_img)

    new_img.save(img_dir + "/example_bundle_apt0.png")

    # planner3dvis.go()



def bundle_apt2_zoomed_baseline():
    img_dir = rmp_io.resolve_directory("figures/report3/example_bundle_apt2_zoomed_baseline")
    base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"

    worldgen = FuseWorldGen(base_path)
    worldgen.set_goal(np.array([ 1.8438590206666579, 3.0334141719669705, 3.2397430776684684 ]))
    
    planner3dvis = Planner3dVis(worldgen, distancefield=False, draw_bbox=False, start=False)

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

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=colors[i], name=f"simple_target{i}", trajectory_size=0.03)
        

    view = CameraBase.View(
        **{
            # This is here for reproducibility. To get initial nice view parameters simply run this script, 
            # get a nice view, and hit ctrl-c, as this will copy the parameters to your clipboard. Paste them in notepad
            # and copy these 4 values into the dict below. (Note that you have to change "front" to "pos")
        "pos" : [ -0.15605288640816245, 0.34140117396383723, -0.92687255599666729 ],
        "lookat" : [ -1.1562862463553545, 2.7203580434272712, 2.0147353151604692 ],
        "up" : [ -0.065468243232714801, 0.9327303745641724, 0.35458138345576024 ],
        "zoom" : 0.25999999999999956
        }
    )


    vc = planner3dvis.plot3d.vis.get_view_control()
    vc.set_lookat(view.lookat)
    vc.set_front(view.pos)
    vc.set_up(view.up)
    vc.set_zoom(view.zoom)
    planner3dvis.plot3d.vis.poll_events()
    planner3dvis.plot3d.vis.update_renderer()
    planner3dvis.plot3d.vis.capture_screen_image(img_dir + "/example_bundle_apt2_zoomed_baseline.png", True)

    # planner3dvis.go()


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    # baseline_vs_ffn()
    # ffn_stuck_wall()
    # rnn_over_wall()
    # rnn_stuck()
    # rnn_main_example_planes(False)
    # rnn_main_example_sphere_box(False)
    # bundle_apt2(images_only=False)
    sun3d_home_at(images_only=False, chomp=True)
    # bundle_apt0(images_only=False)
    # bundle_apt2_zoomed_baseline()

