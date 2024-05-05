
from comparison_planners.rrt.planner_rrt import RRTPlanner


class RRTPlannerTesterWrapper(RRTPlanner):
    # See chomp_tester.py for why we do all this
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._esdf = None
        self._tsdf = None
        self._start = None
        self._goal = None

    @property
    def requires_esdf(self):
        return True
    
    @property
    def requires_geodesic(self):
        return False

    def collided(self):
        # Not really a thing for rrt
        return False

    def diverged(self):
        # Not a thing for rrt
        return False

    def step(self, _unused_step_number=-1):
        if self._esdf is None:
            raise RuntimeError("ESDF is not set")
        if self._tsdf is None:
            raise RuntimeError("TSDF is not set")
        if self._start is None:
            raise RuntimeError("Start is not set")
        if self._goal is None:
            raise RuntimeError("Goal is not set")
        
        self.set_esdf(self._esdf)
        self.set_tsdf(self._tsdf)
        self.plan(self._start, self._goal)

    def setup(self, start, goal, tsdf, esdf, _unused_geodesic=None):
        self._esdf = esdf
        self._tsdf = tsdf
        self._start = start
        self._goal = goal
    
    def success(self):
        return self._planner.success()
    
    def get_trajectory(self):
        path = self._planner.get_path()
        # We return the path 3 times, as the random world tester expects a tuple of (path, path, path)
        # The velocities and accelerations are undefined for rrt, but we include it anyways to make this work with the other code easily
        return path, path, path
