# Created by Teodor Tchalakov (tcha2@illinois.edu), 2022-04-18

# Standard library includes
import copy
from collections import defaultdict
from multiprocessing.sharedctypes import Value

# Internal library includes
from .ocp_transformer import OCPTransformer, PrototypeOCP
from ..costs import LogBarrierCost, InverseBarrierCost, HalfQuadraticBarrierCost

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

BARRIER_COST_DEFAULT_BOUNDS = (0, 100)
BARRIER_COST_DEFAULT_VALUE = 1.0
BARRIER_COST_DEFAULT_LOG = False
class BarrierCostTransformer(OCPTransformer):
    def __init__(self, system):
        self._scales = {}
        self._limits = {} # Key: obs/ctrlname, Value: (limit, upper)
        super().__init__(system, 'BarrierCostTransformer')

    def set_barrier_type(self, barrierType):
        self.fix_hyperparameters(**{'BarrierType': barrierType})
        
    def set_tunable_scale(self, boundedState, lower=BARRIER_COST_DEFAULT_BOUNDS[0],
                                              upper=BARRIER_COST_DEFAULT_BOUNDS[1],
                                              default_value=BARRIER_COST_DEFAULT_VALUE,
                                              log=BARRIER_COST_DEFAULT_LOG):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            # Changing the scale hyperparameter from Constant to default UniformFloatHyperparameter
            cs = self.get_config_space()
            hp = cs.get_hyperparameter(boundedState+'_BarrierScale')
            name = hp.name
            new_hp = CS.UniformFloatHyperparameter(hp.name, lower, upper, default_value, log=log)            
            cs._hyperparameters[name] = new_hp
        else:
            raise ValueError(str(boundedState) + " is not in system")
    def set_fixed_scale(self, boundedState, scale):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            self.fix_hyperparameters(**{boundedState+"_BarrierScale": scale})
        else:
            raise ValueError(str(boundedState) + " is not in system")

    def _get_scales(self):
        cfg = self.get_config()
        scales = dict()
        for hyper_name in cfg:
            if hyper_name == 'BarrierType':
                continue
            name = hyper_name[:-13] # Removing suffix "_BarrierScale" from the hyperparameter name
            scales[name] = cfg[hyper_name]
        return scales

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.CategoricalHyperparameter("BarrierType", ['Log', 'Inverse', 'HalfQuadratic'], default='Log'))
        for name in self.system.observations + self.system.controls:
            hyper = CS.Constant(name+"_BarrierScale", 0.0)
            cs.add_hyperparameter(hyper)
        return cs

    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp, cost=LogBarrierCost)

    def is_compatible(self, ocp):
        return True
    
    def ocp_requirements(self) -> dict:
        return {}

    def __call__(self, ocp):
        new_ocp = copy.deepcopy(ocp)
        scales = self._get_scales()
        cfg = self.get_config()
        if cfg['BarrierType'] == 'Log':
            new_cost = LogBarrierCost(self.system, ocp.get_obs_bounds(), ocp.get_ctrl_bounds(), scales)
        elif cfg['BarrierType'] == 'Inverse':
            new_cost = InverseBarrierCost(self.system, ocp.get_obs_bounds(), ocp.get_ctrl_bounds(), scales)
        elif cfg['BarrierType'] == 'HalfQuadratic':
            new_cost = HalfQuadraticBarrierCost(self.system, ocp.get_obs_bounds(), ocp.get_ctrl_bounds(), scales)
        else:
            raise ValueError(str(cfg['BarrierType']) + " is not a valid barrier type")
        new_ocp.set_cost(new_cost)
        return new_ocp
