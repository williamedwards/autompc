# Created by Teodor Tchalakov (tcha2@illinois.edu), 2022-04-18

# Standard library includes
import copy
from collections import defaultdict

# Internal library includes
from .ocp_transformer import OCPTransformer, PrototypeOCP
from ..costs import LogBarrierCost

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

def construct_default_bounds():
    return (1e-3, 1e4, 1.0, False)

class LogBarrierCostTransformer(OCPTransformer):
    def __init__(self, system):
        super().__init__(system, 'LogBarrierCostTransformer')

        self._scale_bounds = defaultdict(construct_default_bounds) # Key: obsname, Value: (lower, upper, default, log_scale)
        self._limits = {} # Key: obs/ctrlname, Value: (limit, upper)
        self._scale_fixed = {} # Key: obs/ctrlname, Value: limit

    """
        boundedState : String
            Name of observation or control in the system.

        limit : double
            limit value that barrier is placed at.

        upper : boolean
            True if the limit is an upper limit.
            False if the limit is a lower limit.
    """
    def set_limit(self, boundedState, limit, upper):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            self._limits[boundedState] = (limit, upper)
        else:
            raise ValueError(str(boundedState) + " is not in system")

    """
        boundedState : String
            Name of observation or control in the system.

        limit : double
            limit value that barrier is placed at.

        lower_bound : double
            lower bound of configuration space

        upper_bound : double
            upper bound of configuration space

        default : double
            default value of configuration space
    """
    def set_bounds(self, boundedState, lower_bound, upper_bound, default, log=False):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            if(boundedState in self._limits):
                self._scale_bounds[boundedState] = (lower_bound, upper_bound, default, log)
            else:
                raise ValueError(str(boundedState) + " does not have a configured limit use set_limit")
        else:
            raise ValueError(str(boundedState) + " is not in system")

    def set_fixed_scale(self, boundedState, scale):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            if(boundedState in self._limits):
                self._scale_fixed[boundedState] = scale
            else:
                raise ValueError(str(boundedState) + " does not have a configured limit use set_limit")
        else:
            raise ValueError(str(boundedState) + " is not in system")
            

    def _get_hyperparameters(self, label, bounds_dict, fixed_dict):
        hyperparameters = []
        for name in (self.system.controls + self.system.observations):
            if name in fixed_dict or name not in self._limits:
                continue
            limit, upper = self._limits[name]
            upper_string = "Upper"
            if(not upper):
                upper_string = "Lower"
            lower_scale, upper_scale, default, log = bounds_dict[name]
            hyper = CSH.UniformFloatHyperparameter("{}_{}_{}".format(name, upper_string, label),
                    lower=lower_scale, upper=upper_scale, default_value=default, log=log)
            hyperparameters.append(hyper)
        return hyperparameters

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        hypers = self._get_hyperparameters("LogBarrier", self._scale_bounds, self._scale_fixed)
        cs.add_hyperparameters(hypers)
        return cs

    def _get_boundedState(self, cfg, label, fixed_dict):
        boundedStates = dict()
        for name in (self.system.controls + self.system.observations):
            if name in fixed_dict:
                limit, upper = self._limits[name]
                scale = self._scale_fixed[name]
                boundedStates[name] = (limit, scale, upper)
            elif name in self._limits:
                limit, upper = self._limits[name]
                upper_string = "Upper"
                if(not upper):
                    upper_string = "Lower"
                hyper_name = f"{name}_{upper_string}_{label}"
                scale = cfg[hyper_name]
                boundedStates[name] = (limit, scale, upper)
        return boundedStates

    def get_default_config_space(self):
        return CS.ConfigurationSpace()

    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp, cost=LogBarrierCost)

    def is_compatible(self, ocp):
        return True
    
    def ocp_requirements(self) -> dict:
        return {}

    def __call__(self, ocp):
        boundedStates = self._get_boundedState(self.get_config(), "LogBarrier", self._scale_fixed)
        new_cost = LogBarrierCost(self.system, boundedStates)
        new_ocp = copy.deepcopy(ocp)
        new_ocp.set_cost(new_cost)
        return new_ocp

    