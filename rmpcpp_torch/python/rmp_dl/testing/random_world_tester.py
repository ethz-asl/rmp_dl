
import json
import numpy as np
from rmp_dl.learning.lightning_module import RayLightningModule
from rmp_dl.learning.util import WandbUtil
from rmp_dl.learning.model_io.model_util import ModelUtil
from rmp_dl.planner.planner import PlannerRmp
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.planner.planner_params import LearnedPolicyRmpParameters, PlannerParameters, RayObserverParameters, RaycastingCudaPolicyParameters, TargetPolicyParameters
from rmp_dl.util.trajectory import TrajectoryUtils
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import ProbabilisticWorldgenFactory
from rmp_dl.learning.data.seed_manager import WorldgenSeedManager
import rmp_dl.util.io as rmp_io
import pandas as pd
from typing import Any, Callable, List, Optional, Tuple
import time

import functools

import argparse

import multiprocessing as mp
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase
import torch

import wandb
import os

class RandomWorldTester:
    def __init__(self, planner_constructor: Callable[[], PlannerRmp], num_workers: int=1):
        """Does tests with random worlds, monitoring statistics such as success rate, planning time, etc.
        We use a planner constructor callable instead of a planner instance such that we can do multiprocessing,
        as the c++ planner objects are (generally) not picklable. Use functools.partial to pass callables
        that are picklable. https://docs.python.org/3/library/functools.html#functools.partial

        """
        self.planner_constructor = planner_constructor
        self.num_workers = num_workers

    def full_validation(self, runs=100, world_type: str="sphere_box") -> pd.DataFrame:
        return RandomWorldTester.full_validation_static(runs, world_type, self.planner_constructor, self.num_workers)

    @staticmethod
    def full_validation_static(runs, world_type, planner_constructor, num_workers) -> pd.DataFrame:
        step = RandomWorldTester.resolve_steps_for_world_type(world_type)
        df = pd.DataFrame()
        run_data = []
        for obstacles in range(*step):
            for i in range(runs):
                seed = WorldgenSeedManager.get_seed("full_test", obstacles, i)
                run_data.append((obstacles, seed))
        
        if num_workers == 1:
            data = []
            def callback(d):
                print(f"Main has received: {len(data)} / {len(run_data)} runs", flush=True)
                data.append(d)
            RandomWorldTester.do_runs(planner_constructor, run_data, result_callback=callback, world_type=world_type)
            df = pd.concat(data)
        else:
            # We make chunks of the data that multiprocess workers work on. We make more chunks than workers, 
            # so that the load is evenly distributed
            chunks = np.array_split(np.array(run_data), min(num_workers * 10, len(run_data)))

            ctx = mp.get_context('spawn')

            m = ctx.Manager()
            data_queue = m.Queue(num_workers * 10)

            with ctx.Pool(num_workers) as pool:
                result = pool.starmap_async(RandomWorldTester.do_runs, [(planner_constructor, chunk, data_queue.put, world_type) for chunk in chunks])
                
                data = []

                def data_loop():
                    while not data_queue.empty():
                        print(f"Main has received: {len(data)} / {len(run_data)} runs", flush=True)
                        data.append(data_queue.get())

                while not result.ready():
                    data_loop()
                    time.sleep(0.1)

                data_loop()

                if not result.successful():
                    print(result.get())
                    raise RuntimeError("Testing failed")
                
                if len(run_data) != len(data):
                    raise RuntimeError("Testing failed")
                
                df = pd.concat(data)
        return df       

    @staticmethod
    def do_runs(planner_constructor: Callable[[], PlannerRmp], 
                runs: List[Tuple[int, int]], 
                result_callback: Callable[[pd.DataFrame], None], 
                world_type: str):
        planner = planner_constructor()
        for obstacles, seed in runs:
            result_callback(RandomWorldTester.do_single_run(planner, obstacles, seed, world_type))
    
    @staticmethod
    def do_single_run(planner: PlannerRmp, obstacles, seed: int, world_type: str = "sphere_box") -> pd.DataFrame:
        worldgen = RandomWorldTester.resolve_world_type(world_type)(obstacles, seed)
        planner.setup(worldgen.get_start(), worldgen.get_goal(), 
                          worldgen.get_tsdf(), 
                          worldgen.get_esdf() if planner.requires_esdf else None,
                          worldgen.get_distancefield() if planner.requires_geodesic else None)
        
        start = time.time()
        planner.step(-1)
        totaltime = time.time() - start

        return RandomWorldTester.get_df(planner, worldgen, totaltime, obstacles, seed)

    @staticmethod
    def resolve_world_type(world_type: str) -> Callable[[int, int], WorldgenBase]:
        if world_type == "sphere_box":
            return ProbabilisticWorldgenFactory.sphere_box_world
        elif world_type == "planes": 
            return ProbabilisticWorldgenFactory.plane_world
        else:
            raise ValueError(f"Unknown world type: {world_type}")

    @staticmethod
    def resolve_steps_for_world_type(world_type: str) -> Tuple[int, int, int]:
        if world_type == "sphere_box":
            return (0, 210, 10)
        elif world_type == "planes": 
            return (0, 105, 5)
        else:
            raise ValueError(f"Unknown world type: {world_type}")

    @staticmethod
    def get_df(planner: PlannerRmp, worldgen: WorldgenBase, totaltime: float, obstacles: int, seed: int) -> pd.DataFrame:
        """
        Get a dataframe from a planning run
        """
        success = planner.success()
        positions, velocities, accelerations = planner.get_trajectory()
        positions, velocities, accelerations = np.array(positions), np.array(velocities), np.array(accelerations)
        length = TrajectoryUtils.get_length(positions)
        discrete_length = len(positions)
        average_velocity = TrajectoryUtils.get_mean_norm(velocities)
        max_velocity = TrajectoryUtils.get_max_norm(velocities)
        stddev_velocity = TrajectoryUtils.get_std_norm(velocities)
        average_acceleration = TrajectoryUtils.get_mean_norm(accelerations)
        max_acceleration = TrajectoryUtils.get_max_norm(accelerations)
        stddev_acceleration = TrajectoryUtils.get_std_norm(accelerations)
        smoothness = TrajectoryUtils.get_smoothness(positions)
        world_density = worldgen.get_density() 
        distance_to_goal = np.linalg.norm(worldgen.get_goal() - positions[-1])
        collided = planner.collided()
        diverged = planner.diverged()

        data = [obstacles, seed, success, collided, diverged, distance_to_goal, 
             totaltime, length, discrete_length, 
             smoothness,
             average_velocity, max_velocity, stddev_velocity, 
             average_acceleration, max_acceleration, stddev_acceleration, 
             world_density]

        return pd.DataFrame([data], columns = RandomWorldTester._get_columns())

    @staticmethod
    def _get_columns():
        return \
            ["obstacles", "seed", "success", "collided", "diverged", "distance to goal",
             "time", "length", "discrete length", 
             "smoothness",
             "average velocity", "max velocity", "stddev velocity",
             "average acceleration", "max acceleration", "stddev acceleration",
             "world density"
             ]
        


    @staticmethod
    def test_wandb_learned(wandb_id: str, version: str, num_workers, world_type: str, 
                           decoder_type: str, robot_size: float=None):
        """Test a wandb model

        Args:
            wandb_id (str): Wandb id
            version (str): Artifact version on wandb
            num_workers (int, optional): Number of workers for rollouts.
            world_type (str, optional): Planes, sphere_box. 
            decoder_type (str, optional): Type of ray decoder used.
        """
        dirname = os.path.dirname(__file__)
        path_name = os.path.join(dirname, '../../data/wandb')
        os.makedirs(path_name, exist_ok=True)
        wandb.init(project="rmp_dl", entity="rmp_dl", id=wandb_id, resume='must', dir=path_name)
        run = wandb.run
        if run is None: 
            raise ValueError("Init failed")
        
        
        file = WandbUtil.download_model(wandb_id, version, f"data/tmp/models/{wandb_id}-{version}")
        
        raycasting_cuda_params = RaycastingCudaPolicyParameters.from_yaml_general_config(expert=False)

        learned_policy_rmp_params = LearnedPolicyRmpParameters.from_yaml_general_config()
        learned_policy_ray_observer_params = RayObserverParameters.from_yaml_general_config()
        
        planner_params = PlannerParameters.from_yaml_general_config()
        
        constructor = functools.partial(PlannerFactory.learned_planner_from_checkpoint_path, 
                                        file, wandb_id, decoder_type, 
                                        robot_size,
                                        raycasting_cuda_params, 
                                        learned_policy_rmp_params, 
                                        learned_policy_ray_observer_params,
                                        planner_params,
                                        )
        
        data = RandomWorldTester(constructor, num_workers=num_workers).full_validation(world_type=world_type)

        table = wandb.Table(dataframe=data)
        namestr = f"test_table-{wandb_id}-{world_type}"

        aliasstr = f"model{version}"
        if decoder_type is not None:
            aliasstr += f"-{decoder_type}"

        if robot_size is not None:
            aliasstr += f"-rsz:{robot_size}"

        artifact = wandb.Artifact(name=namestr, type="test_table")

        artifact.add(table, f"test_table_{world_type}")
        run.log_artifact(artifact, aliases=[aliasstr])


    @staticmethod 
    def test_other(method_type: str, num_workers: int = 1, world_type="sphere_box", wandb_id=None):
        if method_type == "baseline":
            name = "Baseline"
            target_policy_params = TargetPolicyParameters.from_yaml_general_config()
            raycasting_cuda_policy_parameters = RaycastingCudaPolicyParameters.from_yaml_general_config()
            planner_params = PlannerParameters.from_yaml_general_config()
            
            constructor = functools.partial(PlannerFactory.baseline, target_policy_params, raycasting_cuda_policy_parameters, planner_params)
        elif method_type == "expert":
            name = "Expert"
            target_policy_params = TargetPolicyParameters.from_yaml_general_config()
            # Expert uses slightly different parameters, as it is more safe as it should not go straight into a wall
            raycasting_cuda_policy_parameters = RaycastingCudaPolicyParameters.from_yaml_general_config(expert=True)
            planner_params = PlannerParameters.from_yaml_general_config()

            constructor = functools.partial(PlannerFactory.expert_planner, target_policy_params, raycasting_cuda_policy_parameters, planner_params)
        else:
            raise ValueError(f"Unknown method type: {method_type}")

        dirname = os.path.dirname(__file__)
        path_name = os.path.join(dirname, '../../data/wandb')
        os.makedirs(path_name, exist_ok=True)
        
        if wandb_id is None or wandb_id == "":
            # For expert and baseline it can happen that we have no run logged yet, so we create a new one
            wandb.init(
                # set the wandb project where this run will be logged
                project="rmp_dl",
                name=name,
                dir=path_name,
                
                # track hyperparameters and run metadata
                config={
                    "build_type": "Release",
                    "params": rmp_io.ConfigUtil.get_yaml_general_params()
                },
            )
        if wandb_id is not None:
            # For expert and baseline we can also resume the run, and add e.g. different world_types to the same run
            wandb.init(project="rmp_dl", entity="rmp_dl", id=wandb_id, resume='must', dir=path_name)

        data = RandomWorldTester(constructor, num_workers=num_workers).full_validation(world_type=world_type)

        wandb_id = wandb.run.id

        table = wandb.Table(dataframe=data)
        artifact = wandb.Artifact(name=f"test_table-{wandb_id}-{world_type}", type="test_table")
        artifact.add(table, f"test_table_{world_type}")
        wandb.log_artifact(artifact)

if __name__ == "__main__":
    # Starting tests at the exact time causes some issues with the wandb API when reopening a run, 
    # so we wait a random amount of time 
    # time.sleep(np.random.randint(10, 60))
    parser = argparse.ArgumentParser(description='Run random world tests')
    parser.add_argument('--world_type', type=str, default="sphere_box", help='type of world: \'sphere_box\', \'planes\'')
    parser.add_argument('--decoder_type', type=str, default=None, help='Decoder used if we use a ray output method (instead of cartesian). \
                        Entries in the Halton2dDecoderFactory.resolve_decoder method in halton_decoding.py can be used here. ')
    parser.add_argument('--test_type', type=str, default="learned", help='type of test to run')
    parser.add_argument('--wandb_id', type=str, default=None, help='wandb id of the model to test')
    parser.add_argument('--version', type=str, help='version of the model to test')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers to use')
    parser.add_argument('--partial_ray_observation', type=str, default=None, help='String to set part of the ray observations to a set value, \
                        based on the forward direction. ')

    parser.add_argument('--enable_robot_size', type=bool, default=False, help='Whether we use a size wrapper')
    parser.add_argument('--robot_size', type=float, default=None, help='Size of the robot')

    args = parser.parse_args()

    if not args.enable_robot_size:
        args.robot_size = None

    if args.test_type == "baseline" or args.test_type == "expert":
        # RandomWorldTester.test_other("expert", 1, "planes", "")
        RandomWorldTester.test_other(args.test_type, args.num_workers, args.world_type, args.wandb_id)
    if  args.test_type == "learned":
        # RandomWorldTester.test_wandb_learned("z7a2ciu6", "last", 1, "planes", "max_sum50_decoder", 0.0)
        RandomWorldTester.test_wandb_learned(args.wandb_id, args.version, args.num_workers,
                                             args.world_type, args.decoder_type, args.robot_size)
    
    # planner_constructor = PlannerFactory.simple_target_planner

    # validator = RandomWorldTester(planner_constructor, num_workers=2)
    # result = validator.full_validation(runs=2, step=(0, 300, 100))
    # print(result)


