
from dataclasses import dataclass
from comparison_planners.comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from rmp_dl.worldgenpy.custom_worldgen_factory import CustomWorldgenFactory
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
from chompBindings import ChompParameters, PlannerChomp

from nvbloxLayerBindings import TsdfLayer


@dataclass
class ChompParams:
    w_smooth: float = 0.1
    w_collision: float = 40
    epsilon: float = 0.1
    lmbda: float = 1000
    rel_tol: float = 0.0001
    max_iter: int = 250
    decrease_step_size: bool = True
    map_resolution: float = 0.2
    verbose: bool = True

    def get_cpp_object(self):
        params = ChompParameters()
        params.w_smooth = self.w_smooth
        params.w_collision = self.w_collision
        params.epsilon = self.epsilon
        params.lmbda = self.lmbda
        params.rel_tol = self.rel_tol
        params.max_iter = self.max_iter
        params.decrease_step_size = self.decrease_step_size
        params.map_resolution = self.map_resolution
        params.verbose = self.verbose
        return params

class ChompPlanner:
    def __init__(self, N, params = ChompParams(), cpu=True):
        self.params = params
        self._planner = PlannerChomp(params.get_cpp_object(), N, cpu=cpu)

    def plan(self, start, goal):
        self._planner.plan(start, goal)

    def set_esdf(self, esdf: TsdfLayer):
        # Yes the tsdf is contained in a tsdf layer
        self._planner.set_esdf(esdf)
        # The GPU version is still old and weird and expects a tsdf
        self._planner.set_tsdf(esdf)
    
    def set_tsdf(self, tsdf: TsdfLayer):
        pass

    def get_path(self):
        return self._planner.get_path()

if __name__ == "__main__":
    worldgen = CustomWorldgenFactory.Overhang()
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world(100, seed=12351)
    worldgen = ProbabilisticWorldgenFactory.plane_world(50, seed=1242)

    # planner.set_esdf(worldgen.get_esdf())
    # planner.set_tsdf(worldgen.get_tsdf())
    # planner.plan(worldgen.get_start(), worldgen.get_goal())
    
    planner3dvis_chomp = Planner3dVisComparison(worldgen)

    # planner = ChompPlanner(N=1000)
    # planner3dvis_chomp.add_chomp_planner("chomp", planner, color=[1, 0, 0])

    planner = ChompPlanner(N=200, cpu=True)
    planner3dvis_chomp.add_comparison_planner("chomp", planner, color=[0, 1, 0])

    planner3dvis_chomp.go()


