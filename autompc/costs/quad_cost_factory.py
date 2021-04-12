# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Internal library includes
from .cost_factory import CostFactory
from . import QuadCost

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

class QuadCostFactory(CostFactory):
    def __init__(self, system, goal=None):
        super().__init__(system)
        if goal is None:
            self.goal = None
        else:
            self.goal = goal[:]

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        for obsname in self.system.observations:
            obsgain = CSH.UniformFloatHyperparameter("{}_Q".format(obsname),
                    lower=1e-3, upper=1e4, default_value=1.0, log=True)
            cs.add_hyperparameter(obsgain)
        for obsname in self.system.observations:
            obsgain = CSH.UniformFloatHyperparameter("{}_F".format(obsname),
                    lower=1e-3, upper=1e4, default_value=1.0, log=True)
            cs.add_hyperparameter(obsgain)
        for ctrlname in self.system.controls:
            ctrlgain = CSH.UniformFloatHyperparameter("{}_R".format(ctrlname),
                    lower=1e-3, upper=1e4, default_value=1.0, log=True)
            cs.add_hyperparameter(ctrlgain)
        return cs

    def is_compatible(self, system, task, Model):
        return task.get_cost().has_goal

    def __call__(self, cfg, task, trajs):
        if self.goal is None and task.get_cost().has_goal:
            goal = task.get_cost().get_goal()
        elif self.goal is not None: 
            goal = self.goal
        else:
            raise ValueError("QuadCostFactory requires goal")

        Q = np.zeros((self.system.obs_dim,self.system.obs_dim)) 
        F = np.zeros((self.system.obs_dim,self.system.obs_dim)) 
        R = np.zeros((self.system.ctrl_dim,self.system.ctrl_dim)) 
        for i, obsname in enumerate(self.system.observations):
            Q[i,i] = cfg["{}_Q".format(obsname)]
        for i, obsname in enumerate(self.system.observations):
            F[i,i] = cfg["{}_F".format(obsname)]
        for i, ctrlname in enumerate(self.system.controls):
            R[i,i] = cfg["{}_R".format(ctrlname)]

        return QuadCost(self.system, Q, R, F, goal=goal)
