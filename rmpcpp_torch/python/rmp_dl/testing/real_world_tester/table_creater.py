
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import pandas as pd
from rmp_dl.testing.real_world_tester.real_world_tester import RealWorldTester
import rmp_dl.util.io as rmp_io
import wandb

import asyncio

class TableCreater:
    def __init__(self, worlds: Dict[str, Any], planners: Dict[str, Dict[str, Any]]):
        self.worlds = worlds
        self.planners = planners

        data = self._fetch_data()
        self.df = self._create_table(data)

    def _identify_common_successful_seeds(self):
        # Filter only successful runs
        successful_runs = self.df[self.df['success']]

        # Find common successful seeds for each world
        common_seeds = {}
        for world in successful_runs['world'].unique():
            world_data = successful_runs[successful_runs['world'] == world]
            seeds_by_planner = [set(world_data[world_data['planner'] == planner]['seed']) for planner in world_data['planner'].unique()]
            common_seeds[world] = set.intersection(*seeds_by_planner)
        
        # Create 'Success Subset' flag
        def is_in_success_subset(row):
            if row['success']:
                return row['seed'] in common_seeds.get(row['world'], set())
            return False

        self.df['Success Subset'] = self.df.apply(is_in_success_subset, axis=1)

    def create_summary_table(self):
        self._identify_common_successful_seeds()

        agg_all_df = self.df.groupby(['world', 'planner']).agg({
            'success': 'mean',
            # 'length': ['mean',],
            # 'time': ['mean',],
            # 'smoothness': ['mean', 'std']
        }).reset_index()

        success_df = self.df[self.df["success"]]

        agg_success_df = success_df.groupby(['world', 'planner']).agg(
            length_success_only=('length', 'mean'),
            time_success_only=('time', 'mean'),
            query_time_success_only=('query_time', 'mean')
        ).reset_index()
        
        success_subset_df = self.df[self.df["Success Subset"]]

        agg_success_subset_df = success_subset_df.groupby(['world', 'planner']).agg(
            length_success_subset_only=('length', 'mean'),
            time_success_subset_only=('time', 'mean'),
            query_time_success_subset_only=('query_time', 'mean')
        ).reset_index()

        agg_successes_df = pd.merge(agg_success_df, agg_success_subset_df, on=['world', 'planner'])
        summary_df = pd.merge(agg_all_df, agg_successes_df, on=['world', 'planner'])
        
        return summary_df

    def _create_table(self, data):
        df = pd.DataFrame()
        for planner_name, planner_data in data.items():
            for world_name, world_data in planner_data.items():
                world_data["planner"] = planner_name
                world_data["world"] = world_name
                df = pd.concat([df, world_data], axis=0)

        df["query_time"]  = df["time"] / df["discrete length"]

        return df

    def _fetch_data(self):
        data = {}
        with ThreadPoolExecutor(max_workers=len(self.planners)) as executor:
            # Creating a list of futures
            futures = {executor.submit(self._get_world_data, planner["wandb_id"]): name for name, planner in self.planners.items()}
            for future in as_completed(futures):
                planner_name = futures[future]
                try:
                    data[planner_name] = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    data[planner_name] = {}  # Handle failed fetches

        return data


    def _get_world_data(self, wandb_id):
        data = {}
        for world_name, world_str in self.worlds.items():
            data[world_name] = self._load_data(wandb_id, world_str)

        return data

    def _load_data(self, wandb_id, world_str):
        api = wandb.Api()
        try:
            artifact = api.artifact(
                f"rmp_dl/{RealWorldTester.wandb_project}/results-{wandb_id}-{world_str}:latest")
        except:
            # If the artifact is not found, return an empty dataframe
            return pd.DataFrame()
        table = artifact.get("dataframes")
        df = pd.DataFrame(data=table.data, columns=table.columns)

        return df


if __name__ == "__main__":

    worlds = {
        "Sun 3d": "sun3d_home_at1", 
        "bundlefusion-apt0": "bundlefusion-apt0",
        # "bundlefusion-apt2": "bundlefusion-apt2"
    }

    planners = {
        "RRT m01 t01": {
            "wandb_id": "4b82t4nv",
        },
        "FFN": {
            "wandb_id": "oc25z19e"
        },
        # "RRN": {
        #     "wandb_id": "nyy2h0xg"
        # },
        # "Baseline": {
        #     "wandb_id": "joooqkgk"
        # }
    }

    tc = TableCreater(worlds, planners)
    print(tc.create_summary_table())