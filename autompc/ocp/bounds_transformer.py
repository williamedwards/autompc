# Created by Kris Hauser, (kkhauser@illinois.edu)

import numpy as np
import copy
import ConfigSpace as CS
from ..system import System
from ..costs.zero_cost import ZeroCost
from .ocp import OCP
from .ocp_transformer import OCPTransformer,PrototypeOCP

class DeleteBoundsTransformer(OCPTransformer):
    """
    Factory which preserves the input OCP cost but removes
    state and control bounds.
    """
    def __init__(self, system : System):
        super().__init__(system, "DeleteBounds")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        return cs

    def is_compatible(self, ocp : OCP):
        return ocp.are_obs_bounded
    
    def ocp_requirements(self) -> dict:
        return {'are_obs_bounded':True}

    def __call__(self, ocp : OCP) -> OCP:
        res = copy.deepcopy(ocp)
        res.set_obs_bounds(np.full(self.system.obs_dim,-np.inf),np.full(self.system.obs_dim,np.inf))
        res.set_ctrl_bounds(np.full(self.system.ctrl_dim,-np.inf),np.full(self.system.ctrl_dim,np.inf))
        return res

    def get_prototype(self, config, ocp):
        res = PrototypeOCP(ocp)
        res.are_ctrl_bounded = False
        res.are_obs_bounded = False
        return res


class KeepBoundsTransformer(OCPTransformer):
    """
    Factory which preserves the input OCP bounds but removes the cost.
    """
    def __init__(self, system : System):
        super().__init__(system, "KeepBounds")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        return cs

    def is_compatible(self, ocp : OCP):
        return {'are_obs_bounded':True}
    
    def ocp_requirements(self) -> dict:
        return {'are_obs_bounded':True}

    def __call__(self, ocp : OCP) -> OCP:
        res = copy.deepcopy(ocp)
        res.set_cost(ZeroCost(self.system))
        return res

    def get_prototype(self, config, ocp):
        res = PrototypeOCP(ocp,ZeroCost)
        return res
