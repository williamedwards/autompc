# Created by William Edwards (wre2@illinois.edu), 2021-01-24

from abc import ABC, abstractmethod
from pdb import set_trace

class CostFactory(ABC):
    """
    The CostFactory class constructs cost objects and contains information
    about hyperparameter information.
    """

    def __init__(self, system):
        """
        Consctruct CostFactory.

        Parameters
        ----------
        system : System
            Robot system for cost factory
        """
        self.system = system

    @abstractmethod
    def get_configuration_space(self):
        """
        Returns ConfigurationSpace for cost factory.
        """
        raise NotImplementedError

    # @abstractmethod
    # def is_compatible(self, system, task, Model):
    #     raise NotImplementedError

    @abstractmethod
    def __call__(self, cfg, trajs):
        """
        Build Cost according to configuration.

        Parameters
        ----------
        cfg : Configuration
            Cost hyperparameter configuration

        trajs : List of Trajectory
            Trajectory training set. This is mostly used
            for regularization cost terms and is not required by
            all CostFactories.  If not required, None can be
            passed instead.
        """
        raise NotImplementedError

    def __add__(self, other):
        from .sum_cost_factory import SumCostFactory
        if isinstance(other, SumCostFactory):
            return other.__radd__(self)
        else:
            return SumCostFactory([self, other])
