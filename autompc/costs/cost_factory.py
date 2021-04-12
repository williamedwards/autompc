# Created by William Edwards (wre2@illinois.edu), 2021-01-24

from abc import ABC, abstractmethod
from pdb import set_trace

class CostFactory(ABC):
    @abstractmethod
    def __init__(self, system):
        self.system = system

    @abstractmethod
    def get_configuration_space(self):
        raise NotImplementedError

    @abstractmethod
    def is_compatible(self, system, task, Model):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, cfg, trajs):
        raise NotImplementedError

    def __add__(self, other):
        from .sum_cost_factory import SumCostFactory
        if isinstance(other, SumCostFactory):
            return other.__radd__(self)
        else:
            return SumCostFactory([self, other])
