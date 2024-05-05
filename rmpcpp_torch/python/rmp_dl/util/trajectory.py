import math
import numpy as np

class TrajectoryUtils:
    @staticmethod
    def get_length(trajectory: np.ndarray):
        if len(trajectory) < 1:
            return 0
        return sum(np.linalg.norm(x - y) for x, y in zip(trajectory[:-1], trajectory[1:])) # type: ignore

    @staticmethod
    def get_mean_norm(input: np.ndarray):
        return np.mean(np.linalg.norm(input, axis=1))
    
    @staticmethod
    def get_std_norm(input: np.ndarray):
        return np.std(np.linalg.norm(input, axis=1))

    @staticmethod
    def get_max_norm(input: np.ndarray):
        return np.max(np.linalg.norm(input, axis=1))

    @staticmethod 
    def get_smoothness(positions: np.ndarray):
        if len(positions) < 3: 
            return 1.
        
        A = positions[1:-1] - positions[:-2] # type: ignore
        B = positions[2:] - positions[1:-1] # type: ignore
        smoothness = 0
        for a, b in zip(A, B):
            smoothness += 1 - 1 / np.pi * np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))

        return smoothness / (len(positions) - 2)


        

if __name__ == "__main__":
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    val = TrajectoryUtils.get_length(points)
    assert val == 3.0, val

    val = TrajectoryUtils.get_mean_norm(points) - 1 / 3 - math.sqrt(5) / 3
    assert -1e-6 < val < 1e-6, val
    
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    val = TrajectoryUtils.get_smoothness(points)
    assert val == 1, val

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    val = TrajectoryUtils.get_smoothness(points)
    assert val == 0.5, val