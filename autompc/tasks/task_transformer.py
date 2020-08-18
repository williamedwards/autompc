# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod

class TaskTransformer(ABC):
    def __init__(self, system):
        self.system = system

    @staticmethod
    @abstractmethod
    def get_configuration_space(system):
        """
        Returns the transformer configuration space.
        """
        raise NotImplementedError

    @abstractmethod
    def is_compatible(self, task):
        """
        Returns true if given task can be transformed
        by this transformer.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, task):
        """
        Returns the transformed task.
        """
        raise NotImplementedError
