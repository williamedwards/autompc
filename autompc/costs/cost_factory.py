# Created by William Edwards (wre2@illinois.edu), 2021-01-24

from abc import ABC, abstractmethod
from pdb import set_trace

class CostFactory(ABC):
    @abstractmethod
    def get_configuration_space(self, system, task, Model):
        raise NotImplementedError

    @abstractmethod
    def get_configuration_space(self, system, task, Model):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, system, task, model, trajs, cfg):
        raise NotImplementedError

    def __add__(self, other):
        from .sum_cost_factory import SumCostFactory
        if isinstance(other, SumCostFactory):
            return other.__radd__(self)
        else:
            return SumCostFactory([self, other])
