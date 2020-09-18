# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod

class ControlEvaluator(ABC):
    def __init__(self, system, task, metric):
        self.system = system
        self.task = task
        self.metric = metric

    @abstractmethod
    def __call__(self, pipeline):
        """
        Returns a callable which maps configuration -> score.
        """
        raise NotImplementedError

