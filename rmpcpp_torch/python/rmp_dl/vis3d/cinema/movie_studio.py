from dataclasses import dataclass
import subprocess
import itertools
import os
import queue
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
import numpy as np
from rmp_dl.learning.data.seed_manager import WorldgenSeedManager
from rmp_dl.learning.util import WandbUtil
from rmp_dl.planner.planner_factory import PlannerFactory
from rmp_dl.vis3d.cinema.actors.planner.planner_trajectory_actor import PlannerTrajectoryActor
from rmp_dl.vis3d.cinema.movie_director import MovieDirector
from rmp_dl.vis3d.cinema.scripts.basic_scripts import OutputDecoderVisScript, OutputDecoderVisScriptFollowCam, RayVisScriptStaticCam, TrajectoryVisScriptStaticCam
from rmp_dl.vis3d.cinema.scripts.script_base import Script
from rmp_dl.vis3d.cinema.scripts.transition_script import TransitionScript
from rmp_dl.vis3d.vis3d import Plot3D
from rmp_dl.worldgenpy.probabilistic_worldgen.probabilistic_worldgen_factory import PlaneWorld, ProbabilisticWorldgenFactoryFunction, SphereBoxWorld
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase

import rmp_dl.util.io as rmp_io

class Studio:
    @dataclass
    class RunParams:
        num_obstacles: int
        seed: int  # Gets resolved to a seed for the worldgen using a function based on the num_obstacles and this seed
        worldgen_function: ProbabilisticWorldgenFactoryFunction = SphereBoxWorld()
        script: Script = OutputDecoderVisScript() #TransitionScript(start=100, length=60)
        decoder_type: str = "max_sum512_decoder"
        _start: Optional[np.ndarray] = None # np.array([5., 5., 1.])
        _goal: Optional[np.ndarray] = None # np.array([5., 5., 9.])

        def dirname(self):
            return f"{self.num_obstacles}-{self.seed}-{type(self.worldgen_function).__name__}-{type(self.script).__name__}-{self.decoder_type}"

        @property
        def start(self):
            if self._start is None:
                self._set_start_goal()
            return self._start
        
        @property
        def goal(self):
            if self._goal is None:
                self._set_start_goal()
            return self._goal
        
        def _set_start_goal(self):
            while True:
                self._start = 1 + np.random.rand(3) * 8
                self._goal = 1 + np.random.rand(3) * 8
                if np.linalg.norm(self._start - self._goal) > 9.0:
                    break


    def __init__(self, directory, wandb_id, version):
        self.directory = directory
        self.wandb_id = wandb_id
        self.file = WandbUtil.download_model(wandb_id, version, f"data/tmp/models/{wandb_id}-{version}")

        self.sweeps = []
    

    def add_sweep(self, sweep_name: str, sweep_values: Iterable[Any]):
        """Add sweep over the default run params. Note that sweeps are nested. 
        """
        self.sweeps.append((sweep_name, sweep_values))

    def _create_run_param_list(self) -> List[RunParams]:
        sweep_names, sweep_values = zip(*self.sweeps)
        values_tuple = itertools.product(*sweep_values)

        run_params = []
        for values in values_tuple:
            run_params.append(Studio.RunParams(**dict(zip(sweep_names, values))))
        
        return run_params

    def go(self, num_workers):
        run_params = self._create_run_param_list()

        if num_workers == 1:
            self.single_process(run_params)

        for run_param in run_params:
            pass

    
    def single_process(self,run_params):
        for run_param in run_params:
            Studio._do_run(run_param, self.file, self.wandb_id, self.directory)

    @staticmethod
    def _worker(input_queue: "queue.Queue[Studio.RunParams]", model_file: str, wandb_id: str, output_directory: str):
        while True:
            run_param = input_queue.get()
            if run_param is None:
                return

            Studio._do_run(run_param, model_file, wandb_id, output_directory)


    @staticmethod
    def _do_run(run_param, model_file, wandb_id, output_directory):
        planner = PlannerFactory.learned_planner_with_ray_observer_from_checkpoint_path(model_file, wandb_id, decoder_method=run_param.decoder_type)

        seed = WorldgenSeedManager.get_seed("test", run_param.num_obstacles, run_param.seed)
        worldgen = run_param.worldgen_function(num_obstacles=run_param.num_obstacles, 
                                                seed=seed,
                                                start=run_param.start,
                                                goal=run_param.goal)

        distancefield = worldgen.get_distancefield() if planner.requires_geodesic else None
        esdf = worldgen.get_esdf() if planner.requires_esdf else None
        planner.setup(worldgen.get_start(), worldgen.get_goal(), worldgen.get_tsdf(), esdf, distancefield)
        
        trajectory = PlannerTrajectoryActor(planner, name="planner")
        md = MovieDirector()

        run_param.script(md, trajectory)
        
        world_mesh = Plot3D.get_world_geometry(worldgen)
        goal_geometry = Plot3D.get_sphere_geometry(worldgen.get_goal(), color=np.array([0, 1, 0]))
        md.set_initial_geometries([world_mesh, goal_geometry])

        if output_directory is not None:
            output_directory += "/" + run_param.dirname()
            try:
                os.makedirs(output_directory)
            except FileExistsError:
                print(f"Folder exists for run {run_param.dirname()}, skipping")
                return

        Studio.do_run_and_make_video(md, output_directory, run_param.dirname() + ".mp4")

    @staticmethod
    def do_run_and_make_video(md, output_directory, filename, delete_images=True):
        md.go(output_directory)

        if output_directory is not None:
            Studio._make_video(output_directory, output_name=filename)
            if delete_images:
                Studio._delete_images(output_directory)


    @staticmethod 
    def _make_video(directory, output_name="output.mp4"):

        cmd = f"ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 {output_name}"

        process = subprocess.Popen(cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
    
    @staticmethod
    def _delete_images(directory):
        cmd = f"rm *.png"
        process = subprocess.Popen(cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)

if __name__ == "__main__":
    np.random.seed(0)
    wandb_id = "5msibfu3"
    directory = rmp_io.resolve_directory(f"media/videos/rollouts-{wandb_id}/sphere_box")
    # directory = None
    studio = Studio(directory, wandb_id, "v10")
    studio.add_sweep("script", [TrajectoryVisScriptStaticCam()])
    studio.add_sweep("worldgen_function", [SphereBoxWorld()])
    studio.add_sweep("seed", list(range(40)))
    studio.add_sweep("num_obstacles", list(range(20, 70, 10)))
    studio.add_sweep("decoder_type", ["max_sum50_decoder"])

    studio.go(1)




