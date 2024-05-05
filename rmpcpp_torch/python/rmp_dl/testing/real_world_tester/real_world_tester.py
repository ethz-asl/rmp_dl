from dataclasses import dataclass
import os
import pickle
import shutil
import time
from typing import Callable, List
from comparison_planners.rrt.planner_rrt import RRTPlanner
import numpy as np
import pandas as pd
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.testing.chomp.chomp_tester import ChompPlannerTesterWrapper
from rmp_dl.testing.chomp.rrt.rrt_planner_tester_wrapper import RRTPlannerTesterWrapper
from rmp_dl.testing.random_world_tester import RandomWorldTester
from rmp_dl.testing.real_world_tester.fuse_world_randomizer import WorldgenStartGoalRandomizer, NonCollisionAndMarginPredicate, StartGoalMinDistanceAndNonLineOfSightPredicate
import rmp_dl.util.io as rmp_io
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
import wandb

@dataclass
class TestResult:
    dataframe: pd.DataFrame
    trajectory: np.ndarray

@dataclass
class RunsInfo:
    world_name: str
    planner_str: str
    planner_args: dict
    predicate_factory_str: str

class PredicateFactory:
    @staticmethod
    def resolve_and_add_predicates(worldgen: WorldgenStartGoalRandomizer, predicate_factory_str):
        if predicate_factory_str == "default":
            return PredicateFactory.default_predicates(worldgen)
        
    @staticmethod
    def default_predicates(worldgen: WorldgenStartGoalRandomizer):
        worldgen.add_start_goal_point_predicate_esdf(NonCollisionAndMarginPredicate(margin=0.4))
        worldgen.add_start_goal_point_predicate_tsdf(NonCollisionAndMarginPredicate(margin=0.2))
        worldgen.add_start_goal_point_pair_predicate_tsdf(StartGoalMinDistanceAndNonLineOfSightPredicate(min_distance=3.0))

class AllPlannerFactory:
    @staticmethod
    def resolve_planner_constructor(planner_str):
        if planner_str == "rrt":
            return RRTPlannerTesterWrapper
        if planner_str == "learned":
            return PlannerFactory.learned_planner_from_wandb_id
        if planner_str == "baseline":
            return PlannerFactory.baseline_planner_with_default_params
        if planner_str == "chomp":
            return ChompPlannerTesterWrapper
        else:
            raise ValueError(f"Unknown planner string: {planner_str}")

class WorldPaths:
    mapping = {
        "sun3d_home_at1": "sun3d-home_at-home_at_scan1_2013_jan_1",
        "bundlefusion-apt2": "bundlefusion-apt2",
        "bundlefusion-apt0": "bundlefusion-apt0",
        }
    @staticmethod
    def resolve(world_str: str):
        if world_str in WorldPaths.mapping:
            return rmp_io.get_3dmatch_dir() + "/" + WorldPaths.mapping[world_str]
        else:
            raise ValueError(f"Unknown world string: {world_str}")
        
class WorldResolver:
    @staticmethod
    def resolve(world_name: str):
        if world_name in WorldPaths.mapping:
            return FuseWorldGen(WorldPaths.resolve(world_name))
        elif world_name == "sphere_box120":
            return ProbabilisticWorldgenFactory.sphere_box_world_without_start_goal(120, seed=123)
        

class RealWorldTester:
    num_runs = 100
    wandb_project = "real_world_examples_test_runs"
    
    def __init__(self, world_names: List[str], planner_str, planner_args, wandb_id_resume, name):
        self.world_names = world_names
        self.planner_str = planner_str
        self.planner_args = planner_args
        self.wandb_id_resume = wandb_id_resume
        self.name = name

    def run(self, num_workers: int = 0, wandb=True):
        seeds = list(range(self.num_runs))

        if wandb:
            self._init_wandb()

        for world_name in self.world_names:
            if num_workers == 0:
                results = self._single_process_run(seeds, world_name)
            else:
                raise ValueError("Multiprocessing not supported yet")
            
            if wandb:
                self._log_wandb(results, world_name)
        
        if wandb:
            self._finish_wandb()

    def _init_wandb(self):
        dirname = os.path.dirname(__file__)
        path_name = os.path.join(dirname, '../../data/wandb')
        os.makedirs(path_name, exist_ok=True)
        
        if self.wandb_id_resume is None:
            wandb.init(
                project=self.wandb_project, 
                name=self.name, 
                dir=path_name,
                config={
                    "planner_str": self.planner_str,
                    "planner_args": self.planner_args
                }
            )
        else:
            wandb.init(
                project=self.wandb_project,
                id=self.wandb_id_resume,
                resume="must",
                dir=path_name
            )

    def _finish_wandb(self):
        wandb.finish()

    def _log_wandb(self, results: List[TestResult], world_name):
        artifact = wandb.Artifact(name=f"results-{wandb.run.id}-{world_name}", type="results")

        with rmp_io.TempDirectories(f"/trajectories/{wandb.run.id}-{world_name}") as trajectories_dir:
            self._add_results_to_artifact(artifact, results, trajectories_dir)
            wandb.log_artifact(artifact)

    def _single_process_run(self, seeds, world_name):
        results: List[TestResult] = []
        def result_callback(result: TestResult):
            results.append(result)

        runs_info = RunsInfo(world_name, self.planner_str, self.planner_args, "default")
        RealWorldTester._do_runs(seeds, runs_info, result_callback)
        
        return results

    def _add_results_to_artifact(self, artifact, results, trajectories_dir):
        dataframes = [result.dataframe for result in results]
        dataframes = pd.concat(dataframes)
        dataframe_table = wandb.Table(dataframe=dataframes)

        trajectories = {}
        for result in results:
            trajectories[result.dataframe["seed"].iloc[0]] = result.trajectory

        file = trajectories_dir + "/trajectories.pkl"
        with open(file, "wb") as f:
            pickle.dump(trajectories, f)
        
        artifact.add(dataframe_table, "dataframes")
        artifact.add_file(file, "trajectories.pkl")

    @staticmethod
    def _do_run(seed: int, planner, worldgen_randomizer: WorldgenStartGoalRandomizer):
        worldgen = worldgen_randomizer.randomize_start_goal(seed)

        planner.setup(worldgen.get_start(), worldgen.get_goal(), 
                          worldgen.get_tsdf(), 
                          worldgen.get_esdf() if planner.requires_esdf else None)
        
        start = time.time()
        planner.step(-1)
        totaltime = time.time() - start

        # the -1 is for the number of obstacles, which is undefined for the real world tester
        df = RandomWorldTester.get_df(planner, worldgen, totaltime, -1, seed)
        return TestResult(df, planner.get_trajectory()[0]) # get_trajectory returns a tuple of (pos, vel, acc)
    
    @staticmethod
    def _do_runs(seeds, runs_info: RunsInfo, result_callback: Callable[[TestResult], None]):
        worldgen = WorldResolver.resolve(runs_info.world_name)
        worldgen = WorldgenStartGoalRandomizer(worldgen)
        PredicateFactory.resolve_and_add_predicates(worldgen, runs_info.predicate_factory_str)

        planner = AllPlannerFactory.resolve_planner_constructor(runs_info.planner_str)(**runs_info.planner_args)

        for seed in seeds:
            result = RealWorldTester._do_run(seed, planner, worldgen)
            result_callback(result)



def get_parser():
    import argparse

    # Initialize the main parser
    parser = argparse.ArgumentParser(description='Process input arguments for testing different planners.')

    # world_names argument
    parser.add_argument('--name', required=True, help='Name of the test run')
    parser.add_argument('--world_names', nargs='+', required=True, help='List of world names', 
                        default=["sun3d_home_at1", "bundlefusion-apt2", "bundlefusion-apt0"])

    parser.add_argument('--break', required=False, dest='__dummy', action='store_true', help=argparse.SUPPRESS)

    # wandb_id argument
    parser.add_argument('--wandb_id_resume', default=None, help='wandb ID to resume run from(optional)')

    # Subparsers for different planner strategies
    subparsers = parser.add_subparsers(dest='planner_str', required=True, help='Planner type')

    # Subparser for the RRT planner
    rrt_parser = subparsers.add_parser('rrt', help='RRT planner arguments')
    rrt_parser.add_argument('--time', type=float, required=True, help='Time for RRT planner')
    rrt_parser.add_argument('--margin_to_obstacles', type=float, required=True, help='Margin to obstacles for RRT planner')

    # Subparser for the learned planner
    learned_parser = subparsers.add_parser('learned', help='Learned planner arguments')
    learned_parser.add_argument('--wandb_id', required=True, help='wandb ID for learned planner', )
    learned_parser.add_argument('--alias', required=True, help='Alias for learned planner', )
    learned_parser.add_argument('--decoder_method', required=True, help='Decoder method for learned planner', )
    learned_parser.add_argument('--noise_std', type=float, required=False, help='Noise std for learned planner', default=0.0)

    # Subparser for the baseline planner
    baseline_parser = subparsers.add_parser('baseline', help='Baseline planner does not require specific arguments')
    chomp_parser = subparsers.add_parser('chomp', help='Chomp planner arguments')
    chomp_parser.add_argument("--N", type=int, required=True, help="Number of points")


    return parser 


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # args = parser.parse_args(
    #     [
    #         "--name", "RRT world test",
    #         "--world_names", 
    #         "sphere_box120", # "sun3d_home_at1", "bundlefusion-apt2", "bundlefusion-apt0", 
    #         "--break",
    #         "rrt",
    #         "--time", "0.1", "--margin_to_obstacles", "0.1"
    #      ]
    #      )
    
    name = args.name
    world_names = args.world_names
    wandb_id_resume = args.wandb_id_resume
    planner_str = args.planner_str

    if planner_str == "rrt":
        planner_args = {
            "time": args.time,
            "margin_to_obstacles": args.margin_to_obstacles
        }
    elif planner_str == "learned":
        planner_args = {
            "wandb_id": args.wandb_id,
            "alias": args.alias,
            "decoder_method": args.decoder_method
        }
        if args.noise_std is not None:
            planner_args["ray_noise_params"] = {"noise_std": args.noise_std}
            planner_args["add_ray_noise"] = True
    elif planner_str == "chomp":
        planner_args = {
            "N": args.N,
        }
    elif planner_str == "baseline":
        planner_args = {}
    else:
        raise ValueError(f"Unknown planner string: {planner_str}")

    tester = RealWorldTester(world_names, planner_str, planner_args, wandb_id_resume, name)
    tester.run()

    pass


