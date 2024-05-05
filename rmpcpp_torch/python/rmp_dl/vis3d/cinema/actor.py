

import abc
from typing import List
import numpy as np
import open3d as o3d


class ActorBase(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def has_next_step(self) -> bool: ...

    @abc.abstractmethod
    def next_step(self) -> None: ...    

    @abc.abstractmethod
    def get_geometries_to_remove(self) -> List[o3d.geometry.Geometry]: ...

    @abc.abstractmethod
    def get_geometries_to_add(self) -> List[o3d.geometry.Geometry]: ...

