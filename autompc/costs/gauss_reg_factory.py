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
    def __init__(self):
        pass

    def get_configuration_space(self, system, task, Modle):
        cs = CS.ConfigurationSpace()
        reg_weight = CSH.UniformFloatHyperparameter("reg_weight",
                lower=1e-3, upper=1e4, default_value=1.0, log=True)
        cs.add_hyperparameter(reg_weight)
        return cs

    def is_compatible(self, system, task, Model):
        return True

    def __call__(self, system, task, model, trajs, cfg):
        X = np.concatenate([traj.obs[:,:] for traj in trajs])
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=0)
        Q = cfg["reg_weight"] * la.inv(cov)
        F = np.zeros_like(Q)
        R = np.zeros((system.ctrl_dim, system.ctrl_dim))

        return QuadCost(system, Q, R, F, goal=mean)
