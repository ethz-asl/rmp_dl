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


class Vis:
    @staticmethod
    def vis(model):
        worldgen = CustomWorldgenFactory.Overhang()
        # worldgen = ProbabilisticWorldgenFactory.sphere_box_world(190, seed=119000048)
        # worldgen = ProbabilisticWorldgenFactory.plane_world(100, seed=110000036)
        planner3dvis = Planner3dVis(worldgen, distancefield=False, initial_idx=181, initial_modifier=3)

        Planner3dVisFactory.add_simple_target_planner(planner3dvis, color=[1, 1, 1])

        # model = ModelUtil.load_model("5msibfu3", "latest")
        # model = ModelImporter.load_ffn() # From disk
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory("max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner(planner3dvis, name="comp",
                                                model=model, lstm_color=True,
                                                step_vis=False, ray_vis=True)


        
        model = ModelImporter.load_ffn() # From disk
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory("max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner(planner3dvis, name="FFN",
                                                model=model, color=[0, 0, 1], 
                                                step_vis=False, ray_vis=True)
        model = ModelImporter.load_rnn() # From disk
        model = RayModelDirectionConversionWrapper(model)
        model.set_output_decoder_from_factory("max_sum50_decoder")
        Planner3dVisFactory.add_learned_planner(planner3dvis, name="RNN",
                                                model=model, color=[0, 1, 0], 
                                                step_vis=False, ray_vis=True)
        
        planner3dvis.go()