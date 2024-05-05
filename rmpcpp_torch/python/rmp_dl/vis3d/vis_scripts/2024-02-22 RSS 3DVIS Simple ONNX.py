import os
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis_factory import PlannerComparison3dVisFactory
import numpy as np
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.exporter import ModelExporter
from rmp_dl.learning.onnx_model import OnnxModel
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

    filename = os.path.join(ModelExporter.get_model_dir(), "rnn.onnx")
    onnx_model = OnnxModel(filename)

    Planner3dVisFactory.add_learned_planner_minimal(planner3dvis, name="RNN",
                                            model=onnx_model)
    
    planner3dvis.go()


if __name__ == "__main__":
    rnn_over_wall()

