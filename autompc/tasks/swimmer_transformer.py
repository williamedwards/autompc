# Created by William Edwards (wre2@illinois)

import copy

import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from .task_transformer import TaskTransformer
from .quad_cost import QuadCost


class SwimmerTransformer(TaskTransformer):
    def __init__(self, system, **gains):
        super().__init__(system)
        self.Qgains = np.ones(system.obs_dim)
        self.Rgains = np.ones(system.ctrl_dim)
        self.Fgains = np.ones(system.obs_dim)
        for i, obsname in enumerate(system.observations):
            paramname = "{}_log10Qgain".format(obsname)
            if paramname in gains:
                self.Qgains[i] = 10.0**gains[paramname]
        for i, obsname in enumerate(system.observations):
            paramname = "{}_log10Fgain".format(obsname)
            if paramname in gains:
                self.Fgains[i] = 10.0**gains[paramname]
        for i, ctrlname in enumerate(system.controls):
            paramname = "{}_log10Rgain".format(ctrlname)
            if paramname in gains:
                self.Rgains[i] = 10.0**gains[paramname]
        self.target_velocity = gains["target_velocity"]

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        for obsname in system.observations:
            obsgain = CSH.UniformFloatHyperparameter("{}_log10Qgain".format(obsname),
                    lower=-3.0, upper=4.0, default_value=0.0)
            cs.add_hyperparameter(obsgain)
        for obsname in system.observations:
            obsgain = CSH.UniformFloatHyperparameter("{}_log10Fgain".format(obsname),
                    lower=-3.0, upper=4.0, default_value=0.0)
            cs.add_hyperparameter(obsgain)
        for ctrlname in system.controls:
            ctrlgain = CSH.UniformFloatHyperparameter("{}_log10Rgain".format(ctrlname),
                    lower=-3.0, upper=4.0, default_value=0.0)
            cs.add_hyperparameter(ctrlgain)
        target_velocity = CSH.UniformFloatHyperparameter("target_velocity",
                lower=0.0, upper=5.0, default_value=1.0)
        cs.add_hyperparameter(target_velocity)
        return cs

    def is_compatible(self, task):
        return task.get_cost().is_quad

    def __call__(self, task, trajs):
        newtask = copy.deepcopy(task)
        cost = newtask.get_cost()
        Q, R, F = cost.get_cost_matrices()
        for i in range(len(self.Qgains)):
            Q[i,i] *= self.Qgains[i]
        for i in range(len(self.Rgains)):
            R[i,i] *= self.Rgains[i]
        for i in range(len(self.Fgains)):
            F[i,i] *= self.Fgains[i]
        x0 = np.zeros(self.system.obs_dim)
        x0[5] = self.target_velocity
        newcost = QuadCost(self.system, Q, R, F, x0=x0)
        newtask.set_cost(newcost)
        return newtask


