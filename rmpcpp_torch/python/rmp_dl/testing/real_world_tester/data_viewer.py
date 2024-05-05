


import os
import pickle
import shutil
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from rmp_dl.testing.real_world_tester.real_world_tester import RealWorldTester, WorldPaths, WorldResolver
from rmp_dl.vis3d.planner3dvis import Planner3dVis
from rmp_dl.vis3d.utils import Open3dUtils
from rmp_dl.worldgenpy.fuse_worldgen import FuseWorldGen
import wandb

import rmp_dl.util.io as rmp_io 

class RealWorldDataLoader:
    def __init__(self, wandb_ids, world_name, keys):
        self.wandb_ids: List[str] = wandb_ids
        self.world_name = world_name
        self.keys = keys

    def show(self):
        worldgen = WorldResolver.resolve(self.world_name)

        self.planner3dvis = Planner3dVis(worldgen, start=False, goal=False)
        data = {}
        for wandb_id in self.wandb_ids:
            with rmp_io.TempDirectories(f"/trajectories_view/{wandb_id}-{self.world_name}") as download_dir:
                data[wandb_id] = self._load_data(download_dir, wandb_id)
        
        for wandb_id, key in zip(self.wandb_ids, self.keys):
            self._visualize_data(data[wandb_id], key)
        
        self.planner3dvis.go()
    
    def _visualize_data(self, data, key):
        cmap = plt.get_cmap("viridis")
        rgb_values = cmap(np.linspace(0, 1, len(data)))[:, :3] # Slice to exclude alpha channel       
        geometries = []
        for i, traj in enumerate(data.values()):
            traj_geom = Open3dUtils.get_trajectory_geometry(np.array(traj), color=rgb_values[i], lines_between_points=True)
            geometries.extend(traj_geom)
        
        self.planner3dvis.add_geometry_with_hide_callback(geometries, key)


    def _load_data(self, download_dir, wandb_id):
        api = wandb.Api()
        artifact = api.artifact(f"rmp_dl/{RealWorldTester.wandb_project}/results-{wandb_id}-{self.world_name}:latest")
        file = artifact.get_path("trajectories.pkl").download(root=download_dir)

        with open(file, "rb") as f:
            data = pickle.load(f)
            return data

if __name__ == "__main__":
    wandb_ids = ["bbaumllc", "z9tzov0w", "2drufgtt", "21r9z3vt", "titnf7uw", "89c9y06q", 
                 "t8jgu34c", "6vn155f1", "4vu8487v", "x27irm8e"]
    keys = [ord("R"), ord("F"), ord("L"), ord("B"), ord("C"), ord("S"), 
            ord("1"), ord("2"), ord("3"), ord("4")]
    loader = RealWorldDataLoader(wandb_ids, "bundlefusion-apt2", keys)

    # loader = RealWorldDataLoader(["a6fqf16z"], "sphere_box120", [ord("R")])

    loader.show()