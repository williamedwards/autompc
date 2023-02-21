# Standard library includes
import copy
from collections import defaultdict

# External library includes
import numpy as np
import numpy.linalg as la
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

# Internal library includes
from .ocp_transformer import OCPTransformer,PrototypeOCP
from ..costs import Cost, TrackingCost

def construct_default_bounds():
    return (1e-3, 1e4, 1.0, True)

class TrackingCostTransformer(OCPTransformer):
    """
    Cost Transformer for tracking task. 
    Converts a task tracking cost (with full reference trajectory) to an OCP trackinig cost with finite horizon. 
    """
    def __init__(self, system, cost_transformer):
        super().__init__(system, "TrackingCostTransformeere")
        self.cost_transformer = cost_transformer

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace() 
        return cs

    def is_compatible(self, ocp):
        return isinstance(ocp.get_cost(), TrackingCost)
        
    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp, cost=TrackingCost)

    def __call__(self, ocp, t=0, horizon=50):
        full_len = len(ocp.get_cost().goal)
        new_goal = ocp.get_cost().goal[t:np.clip(t+horizon, t, full_len)]
        new_goal = np.pad(new_goal, pad_width=((0, horizon-len(new_goal)),(0,0)), mode='edge')
        new_cost = TrackingCost(ocp.system, self.cost_transformer(ocp, t, horizon).get_cost(), new_goal)
        new_ocp = copy.deepcopy(ocp)
        new_ocp.set_cost(new_cost)
        return  new_ocp
