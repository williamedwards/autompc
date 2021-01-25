# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Standard library includes
from pdb import set_trace

# Internal library includes
from .cost_factory import CostFactory
from .sum_cost import SumCost
from ..cs_utils import *
from . import QuadCost

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

class SumCostFactory(CostFactory):
    def __init__(self, factories):
        self._factories = factories[:]

    @property
    def factories(self):
        return self._factories[:]

    def get_configuration_space(self, *args, **kwargs):
        cs = CS.ConfigurationSpace()
        for i, factory in enumerate(self.factories):
            _fact_cs = factory.get_configuration_space(*args, **kwargs)
            add_configuration_space(cs,"_sum_{}".format(i), _fact_cs)
        return cs

    def is_compatible(self, *args, **kwargs):
        for factory in self.factories:
            if not factory.is_compatible(*args, **kwargs):
                return False
        return True

    def __call__(self, system, task, model, trajs, cfg):
        costs = []
        for i, factory in enumerate(self.factories):
            fact_cs = factory.get_configuration_space(system, task, model.__class__)
            fact_cfg = fact_cs.get_default_configuration()
            set_subspace_configuration(cfg, "_sum_{}".format(i), fact_cfg)
            cost = factory(system, task, model, trajs, fact_cfg)
            costs.append(cost)
        return sum(costs, start=SumCost(system, []))

    def __add__(self, other):
        if isinstance(other, SumCostFactory):
            return SumCostFactory([*self.factories, *other.factories])
        else:
            return SumCostFactory([*self.factories, other])

    def __radd__(self, other):
        if isinstance(other, SumCostFactory):
            return SumCostFactory([*other.factories, *self.factories])
        else:
            return SumCostFactory([other, *self.factories])
