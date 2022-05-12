# Created by William Edwards (wre2@illinois.edu)

# External library includes
import numpy as np
import ConfigSpace as CS

# Internal library includes
from .optimizer import Optimizer


class ZeroOptimizer(Optimizer):
    """
    The Zero otpimizer is a simple optimizer which always returns
    zero control.

    Parameters:

    Hyperparameters:
    """
    def __init__(self, system):
        super().__init__(system, "Zero")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        return cs

    def set_config(self, config):
        pass

    def is_compatible(self, model, ocp):
        return True

    def step(self, state):
        return np.zeros(self.system.ctrl_dim)

    def get_state(self):
        return dict()

    def set_state(self, state):
        pass
