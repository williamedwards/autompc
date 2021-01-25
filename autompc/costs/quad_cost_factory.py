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
    def __init__(self):
        pass

    def get_configuration_space(self, system, task, Model):
        cs = CS.ConfigurationSpace()
        for obsname in system.observations:
            obsgain = CSH.UniformFloatHyperparameter("{}_Q".format(obsname),
                    lower=1e-3, upper=1e4, default_value=1.0, log=True)
            cs.add_hyperparameter(obsgain)
        for obsname in system.observations:
            obsgain = CSH.UniformFloatHyperparameter("{}_F".format(obsname),
                    lower=1e-3, upper=1e4, default_value=1.0, log=True)
            cs.add_hyperparameter(obsgain)
        for ctrlname in system.controls:
            ctrlgain = CSH.UniformFloatHyperparameter("{}_R".format(ctrlname),
                    lower=1e-3, upper=1e4, default_value=1.0, log=True)
            cs.add_hyperparameter(ctrlgain)
        return cs

    def is_compatible(self, system, task, Model):
        return task.get_cost().has_goal

    def __call__(self, system, task, model, trajs, cfg):
        goal = task.get_cost().has_goal
        Q = np.zeros((system.obs_dim,system.obs_dim)) 
        F = np.zeros((system.obs_dim,system.obs_dim)) 
        R = np.zeros((system.ctrl_dim,system.ctrl_dim)) 
        for i, obsname in enumerate(system.observations):
            Q[i,i] = cfg["{}_Q".format(obsname)]
        for i, obsname in enumerate(system.observations):
            F[i,i] = cfg["{}_F".format(obsname)]
        for i, ctrlname in enumerate(system.controls):
            R[i,i] = cfg["{}_R".format(ctrlname)]

        return QuadCost(system, Q, R, F, goal=goal)

