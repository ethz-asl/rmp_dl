
import abc
from typing import Callable, List
from matplotlib import pyplot as plt
import numpy as np
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

import rmp_dl.util.io as rmp_io

from nvbloxLayerBindings import TsdfLayer, EsdfLayer
from rmpPlannerBindings import NVBloxWorld

import open3d as o3d


class PointValidPredicate(abc.ABC):
    @abc.abstractmethod
    def __call__(self, start: np.ndarray, tsdf: TsdfLayer) -> bool:
        pass

class NonCollisionAndMarginPredicate(PointValidPredicate):
    def __init__(self, margin: float=0.0):
        self.margin = margin

    def __call__(self, start: np.ndarray, tsdf: TsdfLayer) -> bool:
        return not NVBloxWorld.collision_with_unobserved_invalid(start, tsdf, self.margin)

class PointPairValidPredicate(abc.ABC):
    @abc.abstractmethod
    def __call__(self, start: np.ndarray, goal: np.ndarray, tsdf: TsdfLayer) -> bool:
        pass

class StartGoalMinDistanceAndNonLineOfSightPredicate(PointPairValidPredicate):
    def __init__(self, min_distance: float):
        self._min_distance = min_distance

    def __call__(self, start: np.ndarray, goal: np.ndarray, tsdf: TsdfLayer) -> bool:
        if np.linalg.norm(start - goal) < self._min_distance:
            return False

        # Check if start and goal are not in line of sight
        if NVBloxWorld.check_motion(start, goal, tsdf):
            return False
        
        return True

class WorldgenStartGoalRandomizer:
    """Creates a world from a fuse worldgen, by randomizing start and goal locations according
    to predefined rules
    """
    def __init__(self, worldgen: WorldgenBase):
        self._worldgen: WorldgenBase = worldgen
        # Default is do nothing
        self._start_goal_point_predicates_tsdf: List[PointValidPredicate] = []
        self._start_goal_pair_predicates_tsdf: List[PointPairValidPredicate] = []
        
        self._start_goal_point_predicates_esdf: List[PointValidPredicate] = []
        self._start_goal_pair_predicates_esdf: List[PointPairValidPredicate] = []

    def randomize_start_goal(self, seed):
        random_engine = np.random.default_rng(seed)
        world_limits = self._worldgen._settings.world_limits
        
        def get_valid_point():
            while True: 
                point = random_engine.uniform(world_limits[0], world_limits[1], 3)
                if self._start_goal_point_predicate(point):
                    return point

        while True:
            start = get_valid_point()
            goal = get_valid_point()

            if self._start_goal_point_pair_predicate(start, goal):
                break

        self._worldgen.set_start(start)
        self._worldgen.set_goal(goal)
        print(f"Seed: {seed}\r", end="")

        return self._worldgen

    def _start_goal_point_predicate(self, point):
        for predicate in self._start_goal_point_predicates_tsdf:
            if not predicate(point, self._worldgen.get_tsdf()):
                return False
        for predicate in self._start_goal_point_predicates_esdf:
            if not predicate(point, self._worldgen.get_esdf()):
                return False
        return True
    
    def _start_goal_point_pair_predicate(self, start, goal):
        for predicate in self._start_goal_pair_predicates_tsdf:
            if not predicate(start, goal, self._worldgen.get_tsdf()):
                return False
        for predicate in self._start_goal_pair_predicates_esdf:
            if not predicate(start, goal, self._worldgen.get_esdf()):
                return False
        return True

    def add_start_goal_point_predicate_tsdf(self, predicate: PointValidPredicate):
        self._start_goal_point_predicates_tsdf.append(predicate)
    
    def add_start_goal_point_predicate_esdf(self, predicate: PointValidPredicate):
        self._start_goal_point_predicates_esdf.append(predicate)

    def add_start_goal_point_pair_predicate_tsdf(self, predicate: PointPairValidPredicate):
        self._start_goal_pair_predicates_tsdf.append(predicate)

    def add_start_goal_point_pair_predicate_esdf(self, predicate: PointPairValidPredicate):
        self._start_goal_pair_predicates_esdf.append(predicate)


if __name__ == "__main__":
    from rmp_dl.vis3d.planner3dvis import Planner3dVis
    base_path = rmp_io.get_3dmatch_dir() + "/sun3d-home_at-home_at_scan1_2013_jan_1"
    # base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt2"
    # base_path = rmp_io.get_3dmatch_dir() + "/bundlefusion-apt0"

    # worldgen = FuseWorldGen(base_path)
    worldgen = ProbabilisticWorldgenFactory.sphere_box_world_without_start_goal(120, seed=1234)
    start_goal_randomizer = WorldgenStartGoalRandomizer(worldgen)


    start_goal_randomizer.add_start_goal_point_predicate_esdf(NonCollisionAndMarginPredicate(margin=0.4))
    start_goal_randomizer.add_start_goal_point_predicate_tsdf(NonCollisionAndMarginPredicate(margin=0.2))
    start_goal_randomizer.add_start_goal_point_pair_predicate_tsdf(StartGoalMinDistanceAndNonLineOfSightPredicate(min_distance=3.0))

    planner3dvis = Planner3dVis(worldgen, start=False, goal=False)

    N = 1000
    
    cmap = plt.get_cmap("viridis")
    rgb_values = cmap(np.linspace(0, 1, N))[:, :3] # Slice to exclude alpha channel

    start_goal_pairs = []
    colors = []
    for seed, color in zip(range(N), rgb_values):
        start_goal_randomizer.randomize_start_goal(seed)

        start = worldgen.get_start()
        goal = worldgen.get_goal()
        start_goal_pairs.extend([start, goal])
        colors.extend([color, color])


    pcd_geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(start_goal_pairs))
    pcd_geometry.colors = o3d.utility.Vector3dVector(np.array(colors))
    planner3dvis.plot3d.add_geometry(pcd_geometry)

    points = np.array(start_goal_pairs)
    indices = np.stack([np.arange(0, 2*N, 2), np.arange(1, 2*N, 2)], axis=1)
    lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(indices))
    lines.colors = o3d.utility.Vector3dVector(np.array(colors[::2]))
    planner3dvis.plot3d.add_geometry(lines)

    planner3dvis.go()