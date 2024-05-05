


import os
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.radar_data.radar3dvis import Radar3DVis
from rmp_dl.radar_data.radar3dvis_factory import Radar3DVisFactory
from rmp_dl.radar_data.radar_data import RadarData


def test():
    # path = os.path.join('experiment7_urban', '01_urban_night_H_processed.bag')
    # path = os.path.join('experiment7_urban', '02_urban_night_H_processed.bag')
    # path = os.path.join('experiment7_urban', '04_urban_night_H_processed.bag')
    path = os.path.join('experiment7_urban', '05_urban_night_F_processed.bag')
    # path = os.path.join('experiment7', '15_tree_slalom_F_processed.bag')
    radar_data = RadarData.Converted(path)

    radar_vis = Radar3DVis(radar_data)
    
    model = ModelUtil.load_model("g2j8uxxd", "last")
    model.model.set_maximum_ray_length(50)
    
    model = RayModelDirectionConversionWrapper(model)
    model.set_output_decoder_from_factory("max_sum50_decoder")

    Radar3DVisFactory.add_radar(radar_vis, model, "radar", color=[1, 0, 0])

    radar_vis.go()

if __name__ == "__main__":
    test()

  