import glob
import os
import pickle
import shutil
import subprocess
from PIL import Image

from typing import Dict, List, Tuple
import numpy as np
from rmp_dl.learning.data.pipeline.multiprocess_communication import DatasetToRolloutCommunicationTypes, RolloutToDatasetCommunicationTypes
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor

import rmp_dl.util.io as rmp_io

from rmp_dl.worldgenpy.worldgen_base import WorldgenBase
import wandb

class Open3dRendering(PipelineObjectBase):
    def __init__(self, 
                 temporary_storage_path: str, 
                 container_name_or_path: str,  
                 **kwargs):
        """Render the trajectories in open3d and save a 2d image to wandb

        Args:
            every_k_epochs (int): Log every k epochs
            container_name_or_path (str): Container name in case of docker, path to singularity container in case of singularity
                  See the script docker/open3d_vis/run_container_script.sh, which checks first for docker on the system, and then for singularity
                  So if you have both docker and singularity, a singularity path here will not work

        """
        super().__init__(**kwargs)
        self.container_name_or_path = container_name_or_path
        self.temp_directory = temporary_storage_path + f"/open3d_trajectory_logging"
        if MpUtil.is_main_process():
            os.makedirs(self.temp_directory, exist_ok=True)

    def setup(self) -> None:
        super().setup()
        self.log(self.current_epoch)
        MpUtil.barrier()

    def teardown(self):
        if MpUtil.is_main_process():
            shutil.rmtree(self.temp_directory, ignore_errors=True)

    def log(self, epoch):
        self.log_info("Logging open3d trajectories")
        # Because there will be multiple trajectories in the same world that must be logged together, 
        # we need to send the sampler outputs all to the same process before we can do anything. 
        sampler_outputs = [sampler_output for sampler_output in self._get_input()]

        sampler_outputs = MpUtil.gather_on_process(sampler_outputs)

        if not MpUtil.is_main_process():
            return

        # Because we do multiple trajectories in the same world from different starting positions, 
        # we need to keep track of which worlds are the same. 
        # Se we create a mapping from a world to a list of sampler outputs that have the same underlying obstacles, 
        # but may have different start and goal locations. 
        # We have implemented the __hash__ and __eq__ methods in SingleWorldBase such that worlds with the same obstacles (but not necessarily
        # same start and goal locations) are equal.
        # So we can check for this by simply checking if the world is in the mapping.
        world_mapping: Dict[WorldConstructor, List[SamplerOutputBase]] = {}

        sampler_output: SamplerOutputBase
        for sampler_output in sampler_outputs:
            world_constructor = sampler_output.get_world_constructor()
            
            # Append the world to the list of worlds that have the same underlying obstacles
            # Again, __hash__ and __eq__ may be the same, but start and goal locations will be different
            world_mapping.setdefault(world_constructor, []).append(sampler_output)


        output_dir_and_process_list: List[Tuple[str, subprocess.Popen]] = []

        # We now loop over the worlds in the mapping and log them. This is done by calling a subprocess that runs the open3d container (docker or singularity)
        # The function below returns the directory and the process, so we can wait for the process to finish later
        for world_index, (world, observation_list) in enumerate(world_mapping.items()):
            o = self._log_single_world(world, observation_list, world_index, epoch)
            output_dir_and_process_list.append(o)
        
        for world_index, (directory, process) in enumerate(output_dir_and_process_list):
            # wait for the process to finish, and get stdout and stderr
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                self.log_info(f"Open3d process failed with return code {process.returncode}")
                self.log_info(f"Stdout: \n{stdout}")
                self.log_info(f"Stderr: \n{stderr}")
                return

            jpg_files = glob.glob(directory + "/output/*.jpeg")
            images = [wandb.Image(filename, caption=f"Epoch {epoch}, Index {world_index}") for filename in jpg_files]
            names = [os.path.basename(filename).split(".")[0] for filename in jpg_files]

            # Log the images to wandb
            for i, image in enumerate(images):
                self.wandb_log({f"{self.name}-{names[i]}-{world_index}": image}, commit=False)
        
        # Remove the epoch directory TODO: Fix this, with asynchronous logging this does not work. 
        # directory = self.temp_directory + f"/{epoch}"
        # try:
        #     shutil.rmtree(directory, ignore_errors=True)
        # except Exception as e:
        #     self.log_info(f"Could not remove directory {directory} with error {e}")

        
    def _log_single_world(self, world_constructor: WorldConstructor, observation_list: List[SamplerOutputBase], index: int, epoch: int):
        trajectories: List[np.ndarray] = []
        # We sort based on the parameters of the world
        # Hopefully this ensures that trajectories are colored consistently
        # (Due to multiprocessing they are out of order)
        for i, sampler_output_base in enumerate(sorted(observation_list, key=lambda s: str(s.get_world_constructor().__dict__))):
            positions = sampler_output_base.get_observations()["state"]["pos"] 
            trajectories.append(positions)

        # Save camera positions
        camera_views = []
        side_view = world_constructor._get_open3d_camera_settings()  # Tuple of (pos, look_at, up)
        camera_views.append(("side_view", side_view))

        top_view = (np.array([0.1, 15.0, 0.0]), *world_constructor._get_open3d_camera_settings()[1:])  # We replace the position with a position above the world
        camera_views.append(("top_view", top_view))

        # Get the goal
        worldgen_base: WorldgenBase = world_constructor()
        goal = worldgen_base.get_goal()

        # Create the directory
        directory = self.temp_directory + f"/{os.getpid()}/{epoch}/{index}"
        os.makedirs(directory, exist_ok=True)
        output_directory = directory + "/output"
        os.makedirs(output_directory, exist_ok=False)

        # Save the geometries
        self._save_geometries_render_info(trajectories, world_constructor, goal, camera_views, directory)

        # Call a process
        script_path = rmp_io.resolve_directory("../docker/open3d_vis/run_container_script.sh")
        process = subprocess.Popen([script_path, self.container_name_or_path, directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return directory, process

    def _save_geometries_render_info(self, 
                         trajectories: List[List[np.ndarray]], 
                         world_constructor: WorldConstructor, 
                         goal: np.ndarray, 
                         camera_views: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]], 
                         directory: str):
        pickle.dump(trajectories, open(directory + "/trajectories.pkl", "wb"))
        pickle.dump(camera_views, open(directory + "/camera_views.pkl", "wb"))
        pickle.dump(goal, open(directory + "/goal.pkl", "wb"))

        if MpUtil.is_multiprocessed():
            # We can't just call export_to_ply here, as this starts a cuda context, which we don't allow in the dataset
            # workers. So therefore we send it to the rollout worker process
            MpUtil.get_dataset_to_rollout_queue().put((DatasetToRolloutCommunicationTypes.SAVE_PLY, (directory + "/world.ply", world_constructor)))
            if MpUtil.get_rollout_to_dataset_queue().get()[0] != RolloutToDatasetCommunicationTypes.FINISHED_SAVING_TO_PLY:
                raise RuntimeError("Could not save ply")

        else:
            world_constructor().export_to_ply(directory + "/world.ply")
        
        
    @staticmethod 
    def save_image(vis):
        filename = rmp_io.get_temp_data_dir() + "/open3d_trajectory_logging.jpeg"
        vis.capture_screen_image(filename, do_render=True)
        Open3dRendering.optimize_image(filename)

        return filename

    @staticmethod
    def optimize_image(filename):
        img = Image.open(filename)
        if img.size > (1280, 720):
            img = img.resize((1280, 720), Image.LANCZOS) # Some sort of antialiasing filter

        img.save(filename, optimize=True, quality=65)
