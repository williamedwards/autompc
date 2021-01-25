# Created by William Edwards (wre2@illinois.edu), 2021-01-24

from abc import ABC, abstractmethod

class ConstraintFactory(ABC):
    @abstractmethod
    def get_configuration_space(self, system, task, Model):
        raise NotImplementedError

    @abstractmethod
    def is_compatible(self, system, task, Model):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, system, task, cost, model, trajs, cfg):
        raise NotImplementedError


