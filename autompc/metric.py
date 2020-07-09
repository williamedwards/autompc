# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, system):
        self.system = system

    @abstractmethod
    def __call__(self, predictor, traj):
        """
        predictor : start, horizon -> predicted obs
            Expected to cache intermediary states.
        """
        raise NotImplementedError
