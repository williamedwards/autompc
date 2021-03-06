# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Internal library includes
from .cost_factory import CostFactory
from . import QuadCost

# External library includes
import numpy as np
import numpy.linalg as la
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

class GaussRegFactory(CostFactory):
    """
    Cost factory for Gaussian regularization cost. This cost encourages the controller
    to stick close to the distribution of the training set, and is typically used in
    combination with another cost function. The factory returns a quadratic cost
    with :math:`Q= w \\Sigma_x^{-1}` and goal = :math:`\mu_x`.

    Hyperparameters:
     - *reg_weight* (float, Lower: 10^-3, Upper: 10^4): Weight of regularization term.
    """
    def __init__(self, system):
        super().__init__(system)

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        reg_weight = CSH.UniformFloatHyperparameter("reg_weight",
                lower=1e-3, upper=1e4, default_value=1.0, log=True)
        cs.add_hyperparameter(reg_weight)
        return cs

    def is_compatible(self, system, task, Model):
        return True

    def __call__(self, cfg, task, trajs):
        X = np.concatenate([traj.obs[:,:] for traj in trajs])
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=0)
        Q = cfg["reg_weight"] * la.inv(cov)
        F = np.zeros_like(Q)
        R = np.zeros((self.system.ctrl_dim, self.system.ctrl_dim))

        return QuadCost(self.system, Q, R, F, goal=mean)
