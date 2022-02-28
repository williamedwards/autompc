# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Standard library includes
import copy
from collections import defaultdict

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

# Internal library includes
from .ocp_factory import OCPFactory
from .ocp import PrototypeOCP

class IdentityFactory(OCPFactory):
    """
    Factory which preserves the input OCP.
    """
    def __init__(self, system):
        super().__init__(system, "Identity")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        return cs

    def is_compatible(self, ocp):
        return True

    def set_config(self, config):
        self.config = config
        
    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp)

    def __call__(self, ocp):
        return copy.deepcopy(ocp)
