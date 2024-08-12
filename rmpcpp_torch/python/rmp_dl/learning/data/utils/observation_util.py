import numpy as np


class ObservationUtil:
    @staticmethod
    def get_angle_deg(observation) -> float:
        pos = observation["state"]["pos"].flatten()
        goal = observation["info"]["goal"].flatten()
        geodesic = observation["expert_policy"]["geodesic"].flatten()

        rel_pos = goal - pos
        distance = np.linalg.norm(rel_pos)
        rel_pos = rel_pos / distance

        geodesic /= np.linalg.norm(geodesic)
        dot = np.dot(rel_pos, geodesic)
        angle = np.degrees(np.arccos(dot))
        return angle
    
    @staticmethod
    def get_distance(observation) -> float:
        pos = observation["state"]["pos"]
        goal = observation["info"]["goal"]
        rel_pos = goal - pos
        distance = float(np.linalg.norm(rel_pos))
        return distance