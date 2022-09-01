# Created by Teodor Tchalakov (tcha2@illinois.edu), 2022-04-18

# Standard library includes
import copy
from collections import defaultdict
from multiprocessing.sharedctypes import Value

# Internal library includes
from .ocp_transformer import OCPTransformer, PrototypeOCP
from ..costs import LogBarrierCost, InverseBarrierCost

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

BARRIER_COST_DEFAULT_BOUNDS = (0, 100)
BARRIER_COST_DEFAULT_VALUE = 1.0
BARRIER_COST_DEFAULT_LOG = False
class LogBarrierCostTransformer(OCPTransformer):
    def __init__(self, system, barrier_type='log'):
        self._scales = {}
        self._limits = {} # Key: obs/ctrlname, Value: (limit, upper)
        self.barrier_type = barrier_type
        if self.barrier_type == 'log':
            super().__init__(system, 'LogBarrierCostTransformer')
        elif self.barrier_type == 'inverse':
            super().__init__(system, 'InverseBarrierCostTransformer')
        else:
            raise ValueError

    def set_tunable_scale(self, boundedState, lower=BARRIER_COST_DEFAULT_BOUNDS[0],
                                              upper=BARRIER_COST_DEFAULT_BOUNDS[1],
                                              default_value=BARRIER_COST_DEFAULT_VALUE,
                                              log=BARRIER_COST_DEFAULT_LOG):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            # Changing the scale hyperparameter from Constant to default UniformFloatHyperparameter
            cs = self.get_config_space()
            if self.barrier_type == 'log':
                hp = cs.get_hyperparameter(boundedState+'_LogBarrier')
            elif self.barrier_type == 'inverse':
                hp = cs.get_hyperparameter(boundedState+'_InverseBarrier')
            name = hp.name
            new_hp = CS.UniformFloatHyperparameter(hp.name, lower, upper, default_value, log=log)            
            cs._hyperparameters[name] = new_hp
        else:
            raise ValueError(str(boundedState) + " is not in system")
    def set_fixed_scale(self, boundedState, scale):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            if self.barrier_type == 'log':
                self.fix_hyperparameters(**{boundedState+"_LogBarrier": scale})
            elif self.barrier_type == 'inverse':
                self.fix_hyperparameters(**{boundedState+"_InverseBarrier": scale})
        else:
            raise ValueError(str(boundedState) + " is not in system")

    def _get_scales(self):
        cfg = self.get_config()
        scales = dict()
        for hyper_name in cfg:
            name = hyper_name[:-11] # Removing suffix "_LogBarrier" from the hyperparameter name
            scales[name] = cfg[hyper_name]
        return scales

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        for name in self.system.observations + self.system.controls:
            hyper = CS.Constant(name+"_LogBarrier", 0.0)
            cs.add_hyperparameter(hyper)
        return cs

    def get_prototype(self, config, ocp):
        if self.barrier_type == 'log':
            return PrototypeOCP(ocp, cost=LogBarrierCost)
        elif self.barrier_type == 'inverse':
            return PrototypeOCP(ocp, cost=InverseBarrierCost)

    def is_compatible(self, ocp):
        return True
    
    def ocp_requirements(self) -> dict:
        return {}

    def __call__(self, ocp):
        new_ocp = copy.deepcopy(ocp)
        scales = self._get_scales()
        if self.barrier_type == 'log':
            new_cost = LogBarrierCost(self.system, ocp.get_obs_bounds(), ocp.get_ctrl_bounds(), scales)
        elif self.barrier_type == 'inverse':
            new_cost = InverseBarrierCost(self.system, ocp.get_obs_bounds(), ocp.get_ctrl_bounds(), scales)
        new_ocp.set_cost(new_cost)
        return new_ocp
