# Created by William Edwards (wre2@illinois)

import copy

import numpy as np
import numpy.linalg as la
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from .task_transformer import TaskTransformer
from .quad_cost import QuadCost

from pdb import set_trace


class GaussianRegTransformer(TaskTransformer):
    def __init__(self, system, state_reg_weight, ctrl_reg_weight):
        super().__init__(system)
        self.state_reg_weight = 10**state_reg_weight
        self.ctrl_reg_weight = 10**ctrl_reg_weight

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        state_reg_weight = CSH.UniformFloatHyperparameter("state_reg_weight",
                    lower=-4.0, upper=4.0, default_value=0.0)
        ctrl_reg_weight = CSH.UniformFloatHyperparameter("ctrl_reg_weight",
                    lower=-4.0, upper=4.0, default_value=0.0)
        cs.add_hyperparameters([state_reg_weight, ctrl_reg_weight])
        return cs

    def is_compatible(self, task):
        return True

    def __call__(self, task, trajs):
        newtask = copy.deepcopy(task)
        cost = newtask.get_cost()
        X = np.concatenate([traj.obs for traj in trajs])
        U = np.concatenate([traj.ctrls for traj in trajs])
        state_mean = np.mean(X, axis=0)
        ctrl_mean  = np.mean(U, axis=0)
        state_cov  = np.cov(X.T)
        ctrl_cov   = np.cov(U.T)
        if len(state_cov.shape) == 0:
            state_cov = state_cov.reshape((1,1))
        if len(ctrl_cov.shape) == 0:
            ctrl_cov = ctrl_cov.reshape((1,1))
        Q = self.state_reg_weight*la.inv(state_cov)
        R = self.ctrl_reg_weight*la.inv(ctrl_cov)
        new_cost = QuadCost(self.system, Q=Q, R=R, x0=state_mean, u0=ctrl_mean)
        newtask.set_cost(cost + new_cost)
        return newtask
