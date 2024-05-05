
import argparse
import functools
import os
from typing import Optional
from comparison_planners.comparison_planners.chomp.planner_chomp import ChompParams, ChompPlanner
from rmp_dl.testing.random_world_tester import RandomWorldTester

import wandb
import rmp_dl.util.io as rmp_io
import yaml


class ChompPlannerTesterWrapper(ChompPlanner):
    # To be able to reuse the code from the random world tester for the rmp planner, I wrap the chomp planner here, that maps all
    # the functions requried by the random world tester to the chomp planner. The chomp planner has a lot less functionality, 
    # so this maps a lot of stuff to basically zeroes. Hacky, but fastest way of doing this
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
        # If chomp is unsuccessful, it will always return that it collided
        return not self.success()

    def diverged(self):
        # Not a thing for CHOMP
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
        # The velocities and accelerations are undefined for chomp, but we include it anyways to make this work with the rmp code
        return path, path, path
    


class ChompTester:
    @staticmethod
    def start_test(num_workers, name: Optional[str] = None, param_file: Optional[str]=None, wandb_id: Optional[str] = None):
        if param_file is None:
            # Default file is in the ./params folder relative to this file
            param_file = os.path.join(os.path.dirname(__file__), "params", "chomp.yml")
        
        # Load the parameters
        with open(param_file) as file:
            params = yaml.safe_load(file)
        
        if wandb_id is None or wandb_id == "":
            # We start a new run
            if name is None:
                raise RuntimeError("Name must be set if wandb_id is None")
            wandb.init(
                # set the wandb project where this run will be logged
                entity="rmp_dl",
                project="chomp_comparisons",
                name=name,
                #dir=rmp_io.resolve_directory("data/wandb"),
                
                # track hyperparameters and run metadata
                config={
                    "params": params
                },)
        else:
            # Add to existing run
            wandb.init(project="chomp_comparisons", entity="rmp_dl", id=wandb_id, resume="must", dir=rmp_io.resolve_directory("data/wandb"))

        wandb_id = wandb.run.id
        
        planner_constructor = functools.partial(ChompPlannerTesterWrapper, N=params["N"], params=ChompParams(**params["params"]))
        df_sb = RandomWorldTester.full_validation_static(100, "sphere_box", planner_constructor, num_workers=num_workers)
        #df_planes = RandomWorldTester.full_validation_static(100, "planes", planner_constructor, num_workers=num_workers)

        world_types = {
            "sphere_box": df_sb,
            #"planes": df_planes
        }

        for world_type, df in world_types.items():
            # Log the results
            table = wandb.Table(dataframe=df)
            artifact = wandb.Artifact(name=f"test_table-{wandb_id}-{world_type}", type="test_table")
            artifact.add(table, f"test_table_{world_type}")
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CHOMP')
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--wandb_id", "-w", type=str, help="Wandb id. If set to none will create a new wandb entry")
    parser.add_argument("--name", "-n", type=str, default="Testrun", help="Name of the run. Only used if wandb_id is none")
    parser.add_argument("--param_file", "-p", type=str, help="Parameter file to use")
    
    args = parser.parse_args()

    ChompTester.start_test(**args.__dict__)
