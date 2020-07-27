# Created by William Edwards

import numpy as np
import numpy.linalg as la

from ..grapher import Grapher
from ..graph import Graph

class KstepGrapher(Grapher):
    def __init__(self, system, kmax, kstep=1, evalstep=1):
        super().__init__(system)
        self.kmax = kmax
        self.kstep = kstep
        self.evalstep = evalstep

    def __call__(self, model, configuration):
        return KstepGraph(self.system, model, configuration, 
                self.kmax, self.kstep, self.evalstep)

class KstepGraph(Graph):
    def __init__(self, system, model, configuration, kmax, kstep, evalstep):
        super().__init__(system, model, configuration)
        self.kmax = kmax
        self.kstep = kstep
        self.evalstep = evalstep
        self.ks = list(range(1, kmax, kstep))
        self.sumsqes = np.zeros((len(self.ks),))
        self.nevals = np.zeros((len(self.ks),))

    def add_traj(self, predictor, traj, training=False):
        for i, k in enumerate(self.ks):
            for j in range(0, len(traj)-k, self.evalstep):
                pred_obs = predictor(j, k)
                actual_obs = traj[j+k].obs
                self.sumsqes[i] += la.norm(pred_obs - actual_obs)**2
                self.nevals[i] += 1

            
    def __call__(self, fig):
        ax = fig.gca()
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Prediction Error (RMSE)")
        rmses = []
        for sumsqe, n in zip(self.sumsqes, self.nevals):
            rmses.append(np.sqrt(sumsqe / n))
        ax.plot(self.ks, rmses)
