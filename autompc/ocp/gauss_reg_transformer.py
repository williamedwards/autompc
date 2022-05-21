# Created by William Edwards (wre2@illinois.edu), 2021-01-24

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
from ..costs.quad_cost import QuadCost

def construct_default_bounds():
    return (1e-3, 1e4, 1.0, True)

class GaussRegTransformer(OCPTransformer):
    """
    Cost Transformer for Gaussian regularization cost. This cost encourages the controller
    to stick close to the distribution of the training set, and is typically used in
    combination with another cost function. The Transformer returns a quadratic cost
    with :math:`Q= w \\Sigma_x^{-1}` and goal = :math:`\mu_x`.

    Hyperparameters:
     - *reg_weight* (float, Lower: 10^-3, Upper: 10^4): Weight of regularization term.
    """
    def __init__(self, system):
        super().__init__(system, "GaussRegTransformer")
        self._mean = None
        self._cov = None

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        reg_weight = CSH.UniformFloatHyperparameter("reg_weight",
                lower=1e-3, upper=1e4, default_value=1.0, log=True)
        cs.add_hyperparameter(reg_weight)
        return cs

    def is_compatible(self, ocp):
        return True

    def set_config(self, config):
        self.config = config
        
    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp, cost=QuadCost)

    def train(self, trajs):
        X = np.concatenate([traj.obs[:,:] for traj in trajs])
        self._mean = np.mean(X, axis=0)
        self._cov = np.cov(X, rowvar=0)
        self.is_trained = True

    def __call__(self, ocp):
        if self._mean is None or self._cov is None:
            raise RuntimeError("GaussRegTransformer must be trained before calling.")

        Q = self.config["reg_weight"] * la.inv(self._cov)
        F = np.zeros_like(Q)
        R = np.zeros((self.system.ctrl_dim, self.system.ctrl_dim))

        new_cost = QuadCost(self.system, Q, R, F, goal=self._mean)
        new_ocp = copy.deepcopy(ocp)
        new_ocp.set_cost(new_cost)

        return  new_ocp
