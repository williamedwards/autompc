# Created by William Edwards (wre2@illinois.edu), 2021-01-25

# Standard library includes
import copy
from pdb import set_trace

# Internal library includes
from .utils.cs_utils import *
from .sysid.model import ModelFactory, Model
from .control.controller import Controller, ControllerFactory
from .costs.cost import BaseCost
from .costs.cost_factory import CostFactory

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

class Pipeline:
    def __init__(self, system, *components):
        self.system = system
        self.model = None
        self.model_factory = None
        self.controller = None
        self.controller_factory = None
        self.cost = None
        self.cost_factory = None

        for component in components:
            if isinstance(component, Model):
                if self.model or self.model_factory:
                    raise ValueError("Pipeline cannot contain multiple models or "
                            + "model factories.")
                self.model = component
            if isinstance(component, ModelFactory):
                if self.model or self.model_factory:
                    raise ValueError("Pipeline cannot contain multiple models or "
                            + "model factories.")
                self.model_factory = component
            if isinstance(component, Controller):
                if self.controller or self.controller_factory:
                    raise ValueError("Pipeline cannot contain multple controllers or "
                            + "controller factories.")
                self.controller = component
            if isinstance(component, ControllerFactory):
                if self.controller or self.controller_factory:
                    raise ValueError("Pipeline cannot contain multple controllers or "
                            + "controller factories.")
                self.controller_factory = component
            if isinstance(component, BaseCost):
                if self.cost or self.cost_factory:
                    raise ValueError("Pipeline cannot contain multple costs or "
                            + "cost factories.")
                self.cost = component
            if isinstance(component, CostFactory):
                if self.cost or self.cost_factory:
                    raise ValueError("Pipeline cannot contain multple costs or "
                            + "cost factories.")
                self.cost_factory = component

        if not (self.model or self.model_factory):
            raise ValueError("Pipeline must contain model or model factory")
        if not (self.controller or self.controller_factory):
            raise ValueError("Pipeline must contain controller or controller factory")
        if not (self.cost or self.cost_factory):
            raise ValueError("Pipeline must contain cost or cost factory")

    def is_compatible(self, system, task):
        return True

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        if self.model_factory:
            model_cs = self.model_factory.get_configuration_space()
            add_configuration_space(cs, "_model", model_cs)
        if self.controller_factory:
            controller_cs = self.controller_factory.get_configuration_space()
            add_configuration_space(cs, "_ctrlr", controller_cs)
        if self.cost_factory:
            cost_factory_cs = self.cost_factory.get_configuration_space()
            add_configuration_space(cs, "_cost", cost_factory_cs)

        return cs

    def __call__(self, cfg, task, trajs):
        # First instantiate and train the model
        if self.model:
            model = self.model
        else:
            model_cs = self.model_factory.get_configuration_space()
            model_cfg = model_cs.get_default_configuration()
            set_subspace_configuration(cfg, "_model", model_cfg)
            model = self.model_factory(model_cfg, trajs)

        # Then create the objective function
        if self.cost:
            cost = self.cost
        else:
            cost_cs = self.cost_factory.get_configuration_space()
            cost_cfg = cost_cs.get_default_configuration()
            set_subspace_configuration(cfg, "_cost", cost_cfg)
            cost = self.cost_factory(cost_cfg, task, trajs)

        new_task = copy.deepcopy(task)
        new_task.set_cost(cost)

        # Then initialize the controller
        if self.controller:
            controller = self.controller
        else:
            controller_cs = self.controller_factory.get_configuration_space()
            controller_cfg = controller_cs.get_default_configuration()
            set_subspace_configuration(cfg, "_controller", controller_cfg)
            controller = self.controller_factory(controller_cfg, new_task, model)

        return controller, task, model
