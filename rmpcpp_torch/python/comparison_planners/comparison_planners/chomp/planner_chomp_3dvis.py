
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.utils import Open3dUtils

import time

class Planner3dVisComparison(Planner3dVis):
    def add_comparison_planner(self, name, planner, color=[1, 0, 0]):
        if name in self.params:
            raise RuntimeError("Name already exists")

        start = time.time()
        planner.set_esdf(self.worldgen.get_esdf())
        planner.set_tsdf(self.worldgen.get_tsdf())
        print("Setting up planner took", time.time() - start, "seconds")

        start = time.time()
        planner.plan(self.worldgen.get_start(), self.worldgen.get_goal())
        print("Planning took", time.time() - start, "seconds")
        trajectory = planner.get_path()
        
        self.trajectory_geometry[name] = Open3dUtils.get_trajectory_geometry(trajectory, color=color, lines_between_points=True)
        if isinstance(self.trajectory_geometry[name], list):
            for geom in self.trajectory_geometry[name]:
                self.plot3d.vis.add_geometry(geom)
        else:
            self.plot3d.vis.add_geometry(self.trajectory_geometry[name])
        self.update_step_geometries()