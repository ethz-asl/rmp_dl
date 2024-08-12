import abc
from typing import Any, Tuple
import numpy as np


class LocationSampler(abc.ABC):
    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)

    @abc.abstractmethod
    def sample(self, *, goal=None, limits=None): ...

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self): ...


class NormalRadiusSampler(LocationSampler):
    def __init__(self, band_mean, band_std, seed):
        super().__init__(seed)
        self.mean = band_mean
        self.std = band_std

    def sample(self, *, goal, limits: Tuple[np.ndarray, np.ndarray]):
        self.goal = goal
        self.limits = limits
        return self

    def __next__(self):
        # First we sample a point on the unit sphere
        # Easiest way is just to generate 3 random numbers from a normal distribution and normalize them
        vec = self.generator.standard_normal(size=3)
        vec /= np.linalg.norm(vec, axis=0)

        radius = self.generator.normal(loc=self.mean, scale=self.std)
        
        point = self.goal + vec * radius

        # Check if the point is within the limits
        if np.all(point > self.limits[0]) and np.all(point < self.limits[1]):
            return point

        return self.__next__()

class UniformWorldLimitsSampler(LocationSampler):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, *, limits, goal=None):
        # Yeha, this is a bit of a hack, goal is there to make the interface consistent 
        # These sampling classes are a mess, and the position sampler class too. TODO: Clean all this up
        self.world_limits = limits
        return self

    def __next__(self):
        return self.generator.uniform(self.world_limits[0], self.world_limits[1])


class StartGoalMinDistSampler(LocationSampler):
    def __init__(self, seed):
        super().__init__(seed)
        self.uniform_sampler = UniformWorldLimitsSampler(seed)

    def sample(self, min_dist, world_limits):
        self.min_dist = min_dist
        self.world_limits = world_limits
        return self

    def __next__(self):
        world_limit_sampler = self.uniform_sampler.sample(goal=None, limits=self.world_limits)

        if not np.linalg.norm(np.array(self.world_limits[1]) - np.array(self.world_limits[0])) > self.min_dist * 1.05:
            raise RuntimeError("Distance between start and goal is too large for the world size")

        while True: 
            start = next(world_limit_sampler)
            goal = next(world_limit_sampler)

            if np.linalg.norm(start - goal) > self.min_dist:
                return start, goal


if __name__ == "__main__":
    sampler = NormalRadiusSampler(3, 0.7)
    sampler.set_seed(1)

    i = 0
    samples = []
    goal = np.array([4.0, 5.0, 6.0])
    for sample in sampler.sample(goal=goal, limits=(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]))):
        print(sample)
        samples.append(sample)
        if (i := i + 1) > 10000:
            break

    mean = np.mean(samples, axis=0)
    print("mean: ", mean)
    assert np.allclose(mean, goal, atol=0.1), mean