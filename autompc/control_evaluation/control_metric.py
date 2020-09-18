# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod

class ControlMetric(ABC):
    def __init__(self, system, task):
        self.system = system
        self.task = task

    @abstractmethod
    def __call__(self, controller, sim_model):
        """
        Returns controller score.
        """
        raise NotImplementedError
