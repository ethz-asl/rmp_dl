from rrtBindings import PlannerRRT, RRTParameters


class RRTPlanner:
    def __init__(self, time=0.5, margin_to_obstacles=0.0):
        self.time = time
        params = RRTParameters()
        params.margin_to_obstacles = margin_to_obstacles
        self._planner = PlannerRRT(params)

    def plan(self, start, goal):
        self._planner.plan(start, goal, self.time)

    def get_path(self):
        return self._planner.get_path()
    
    def set_esdf(self, esdf):
        self._planner.set_esdf(esdf)
    
    def set_tsdf(self, tsdf):
        pass # Not really used
    
    def success(self):
        return self._planner.success()
