# Created by William Edwards (wre2@illinois.edu), 2022-10-24

# Standard library includes
from typing import Optional, List
from abc import ABC, abstractmethod

# Internal project includes
from ..system import System
from ..trajectory import Trajectory

class ModelingBenchmark(ABC):
    def __init__(self, name: str, system: System):
        self.name = name
        self.system = system

    @abstractmethod
    def get_trajs(self, n_trajs: Optional[int] = None) -> List[Trajectory]:
        """
        Returns set of available trajectories for benchmark.

        Parameters
        ----------
        n_trajs : Number of trajectories to access. If None,
                    returns default or maximum number of 
                    trajectories.  Raises excepiton if
                    num_trajs exceeds to self.max_num_trajs.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_num_trajs(self) -> Optional[int]:
        """
        Maximum number of trajectories available for benchmark.
        None if there is no maximum number of trajectories.
        """
        raise NotImplementedError