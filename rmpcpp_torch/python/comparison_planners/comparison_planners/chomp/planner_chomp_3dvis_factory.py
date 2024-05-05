

from comparison_planners.chomp.planner_chomp import ChompPlanner
from comparison_planners.chomp.planner_chomp_3dvis import Planner3dVisComparison
from comparison_planners.rrt.planner_rrt import RRTPlanner


class PlannerComparison3dVisFactory:
    @staticmethod
    def add_chomp_planner(planner3dvis_comparison: Planner3dVisComparison, color=[1, 0, 0], trajectory_size=None, name="chomp"):
         planner = ChompPlanner(N=200)
         planner3dvis_comparison.add_comparison_planner(name, planner, color=color)

    @staticmethod
    def add_rrt_planner(planner3dvis_comparison: Planner3dVisComparison, color=[1, 0, 0], trajectory_size=None, name="rrt"):
         planner = RRTPlanner()
         planner3dvis_comparison.add_comparison_planner(name, planner, color=color)


    