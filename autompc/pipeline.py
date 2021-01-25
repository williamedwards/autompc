# Created by William Edwards (wre2@illinois.edu), 2021-01-25

# Standard library includes
from pdb import set_trace

# Internal library includes
from .cs_utils import *

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

class Pipeline:
    def __init__(self, Model, Controller, cost_factory, constraint_factory):
        self.Model = Model
        self.Controller = Controller
        self.cost_factory = cost_factory
        self.constraint_factory = constraint_factory

    def is_compatible(self, system, task):
        # TODO
        return True

    def get_configuration_space(self, system, task):
        # TODO Fix task properties
        cs = CS.ConfigurationSpace()
        model_cs = self.Model.get_configuration_space(system)
        controller_cs = self.Controller.get_configuration_space(system, task, self.Model)
        cost_factory_cs = self.cost_factory.get_configuration_space(system, task, 
                self.Model)
        add_configuration_space(cs, "_model", model_cs)
        add_configuration_space(cs, "_ctrlr", controller_cs)
        add_configuration_space(cs, "_cost", cost_factory_cs)
        if not self.constraint_factory is None:
            constraint_factory_cs = self.constraint_factory.get_configuration_space(
                    system, task, self.Model)
            add_configuration_space(cs, "_constraint", constraint_factory_cs)

        return cs
