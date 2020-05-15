# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from ..controller import Controller
from ..hyper import IntRangeHyperparam

class ExampleController(Controller):
    def __init__(self, system, model):
        super().__init__(system, model)
        self.horizon = IntRangeHyperparam((1, 100), default_value=10)

    def run(self, traj, latent=None):
        # Implement control logic here

        return u, None

    def run_diff(self, traj, us, latent=None):
        # If controller is differentiable, 
        # return control with gradient here.

        return xnew, None, grad
