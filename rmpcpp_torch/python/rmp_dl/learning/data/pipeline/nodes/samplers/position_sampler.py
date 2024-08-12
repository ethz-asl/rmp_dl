from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
import numpy as np
from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_base import SamplerBase
from rmp_dl.learning.data.pipeline.nodes.samplers.sampling_function import SamplingFunction, SamplingStatistics
from rmp_dl.learning.data.pipeline.nodes.worlds.world_constructor import WorldConstructor
from rmp_dl.learning.data.utils.location_sampler import LocationSampler, NormalRadiusSampler, UniformWorldLimitsSampler
from rmp_dl.planner.planner_params import RayObserverParameters
from rmp_dl.planner.observers.ray_observer import RayObserver
from rmp_dl.worldgenpy.distancefield_gen import DistanceFieldGen
from rmp_dl.worldgenpy.worldgen_base import WorldgenBase


class PositionSampler(SamplerBase):
    """Densely samples random positions in a world
    This class is a bit of a mess honestly, the samplers are strangely defined and it is just a bit chaotic. 
    Also if you use this sampler to sample across multiple epochs, the seed for a worker will be the same across epochs. 
    So the sampling locations will be the same. Though this is mitigated by the fact that the obstacles will be in different locations, 
    so different samples are thrown away each epoch. 
    This does not happen though, as it does not make much sense to use this function across epochs, as this is mainly used for dagger, 
    in which case we use the rollout_sampler which does not have this problem (as it just does a rollout, not a sampling of positions)

    Args:
        SamplerBase (_type_): _description_
    """
    def __init__(self, 
                 num_samples: int, 
                 sampler_type: str, 
                 sampler_params: Dict[str, Any],
                 ray_observer_params: Union[Dict[str, Any], RayObserverParameters],
                 min_fraction_valid: float = 0.0,
                 min_fraction_reachable: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.sampler_type = sampler_type
        self.sampler_params = sampler_params
        self.ray_observer_params = ray_observer_params if type(ray_observer_params) is RayObserverParameters \
            else RayObserverParameters(**ray_observer_params) # type: ignore
        self.sampler = None  # This depends on process ID (seeding), so will be set in setup()
        self.min_fraction_valid = min_fraction_valid
        self.min_fraction_reachable = min_fraction_reachable

    def setup(self):
        # We add the seed of the process to the seed of the sampler
        # We can only do that here and not in __init__, as __init__ happens in the main process and this object
        # gets copied to a subprocess
        self.sampler = self._resolve_sampler(self.sampler_type, self.sampler_params, self.seed)
        super().setup()

    def _resolve_sampler(self, sampler_type: str, sampler_params: Dict[str, Any], seed) -> LocationSampler:
        if sampler_type == "band_sampler":
            return NormalRadiusSampler(**sampler_params, seed=seed)
        elif sampler_type == "uniform_sampler":
            return UniformWorldLimitsSampler(**sampler_params, seed=seed)
        else:
            raise ValueError(f"Unknown sampler type {sampler_type}")
        
    class PositionSamplingFunction(SamplingFunction):
        def __init__(self, num_samples, ray_observer_params, sampler, min_fraction_valid, min_fraction_reachable):
            super().__init__()
            self.num_samples = num_samples
            self.ray_observer_params = ray_observer_params
            self.sampler: LocationSampler = sampler
            self.min_fraction_valid = min_fraction_valid
            self.min_fraction_reachable = min_fraction_reachable


        def _resolve_worldgen_and_distancefield(self, world_constructor: WorldConstructor) -> Tuple[WorldgenBase, DistanceFieldGen]:
            i = 0
            while True:
                i += 1
                worldgen: WorldgenBase = world_constructor(continued_generation=True)
                inflation = world_constructor.inflation

                distancefield = worldgen.get_distancefield(inflation=inflation)

                valid, reachable = distancefield.fraction_valid_and_reachable()
                if valid >= self.min_fraction_valid and reachable >= self.min_fraction_reachable:
                    break

            return worldgen, distancefield, i


        def __call__(self, world_constructor: WorldConstructor) -> Tuple[List[dict], SamplingStatistics, str]:
            worldgen, distancefield, tries = self._resolve_worldgen_and_distancefield(world_constructor)

            observations = []
            valid_count = 0
            
            vs = worldgen.get_voxel_size()
            sample_limits = (worldgen.get_world_limits()[0] + vs, worldgen.get_world_limits()[1] - vs)
            for location in self.sampler.sample(goal=worldgen.get_goal(), limits=sample_limits):
                
                # If we are inside an (inflated) obstacle we continue
                if distancefield.get_esdf_interpolate(location) < world_constructor.inflation: 
                    continue
                
                # If the geodesic is 0, we are inside an unreachable part of the world, so we continue
                # (doing geodesic == 0 also captures the above case, but I think the above check is more safe on boundaries)
                if distancefield.get_distancefield_interpolate(location) == 0.0:
                    continue

                rays = RayObserver.get_rays(worldgen.get_tsdf(), location, self.ray_observer_params, cpu=True).numpy()
                
                if np.linalg.norm(location - worldgen.get_goal()) < worldgen.get_voxel_size() * 2:
                    # Close to the goal the geodesic may be a bit off, so we use the goal vector instead
                    geodesic = worldgen.get_goal() - location
                    geodesic = geodesic / np.linalg.norm(geodesic)
                else:
                    geodesic = distancefield.get_gradient_interpolate(location)

                observation = {
                    "state": {
                        "pos": location,
                        "vel": np.zeros(3),
                        "acc": np.zeros(3),
                    },
                    "expert_policy": {
                        "geodesic": geodesic.flatten(),
                    },
                    "info": {
                        "goal": worldgen.get_goal().copy(),
                        "start": worldgen.get_start().copy(),  # Doesnt matter much this one. Just easier for some plotting things that there is also a start
                        "geodesic_inflation": np.array(world_constructor.inflation),
                    },
                    # Slightly weird naming, stems from how the policies set these values during rollout.
                    "rays": {
                        "rays": rays,
                    },
                }
                
                observations.append(observation)

                if (valid_count := valid_count + 1) >= self.num_samples:
                    break
            
            info_str = f"Worldgen tries: {tries}"
            return observations, SamplingStatistics(), info_str

    def _get_sampling_function(self) -> SamplingFunction:
        return PositionSampler.PositionSamplingFunction(
            num_samples = self.num_samples,
            ray_observer_params = self.ray_observer_params,
            sampler = self.sampler,
            min_fraction_valid = self.min_fraction_valid,
            min_fraction_reachable = self.min_fraction_reachable
            )
            