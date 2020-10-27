# Created by William Edwards

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ..grapher import Grapher
from ..graph import Graph

class KstepGrapher(Grapher):
    def __init__(self, system, kmax, kstep=1, evalstep=1, logscale=False):
        super().__init__(system)
        self.kmax = kmax
        self.kstep = kstep
        self.evalstep = evalstep
        self.logscale = logscale

    def __call__(self, model, configuration):
        return KstepGraph(self.system, model, configuration, 
                self.kmax, self.kstep, self.evalstep, self.logscale)

class KstepGraph(Graph):
    def __init__(self, system, model, configuration, kmax, kstep, evalstep, logscale):
        super().__init__(system, model, configuration)
        self.kmax = kmax
        self.kstep = kstep
        self.evalstep = evalstep
        self.ks = list(range(1, kmax, kstep))
        self.sumsqes = np.zeros((len(self.ks),))
        self.sumsqes_baseline = np.zeros((len(self.ks),))
        self.nevals = np.zeros((len(self.ks),))
        self.logscale = logscale

    def add_traj(self, predictor, traj, training=False):
        for i, k in enumerate(self.ks):
            for j in range(0, len(traj)-k, self.evalstep):
                pred_obs = predictor(j, k)
                actual_obs = traj[j+k].obs
                self.sumsqes[i] += la.norm(pred_obs - actual_obs)**2
                self.sumsqes_baseline[i] += la.norm(traj[j].obs - actual_obs)**2
                self.nevals[i] += 1

    def get_rmses(self):
        rmses = []
        rmses_baseline = []
        for sumsqe, sumsqe_baseline, n in zip(self.sumsqes, 
                self.sumsqes_baseline, self.nevals):
            rmses.append(np.sqrt(sumsqe / n))
            rmses_baseline.append(np.sqrt(sumsqe_baseline / n))
        return rmses, rmses_baseline

            
    def __call__(self, fig):
        ax = fig.gca()
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Prediction Error (RMSE)")
        if self.logscale:
            ax.set_yscale("log")
        rmses, rmses_baseline = self.get_rmses()
        horizs = np.array(self.ks) * self.system.dt
        ax.plot(horizs, rmses, "b-")
        ax.plot(horizs, rmses_baseline, "r-")
        ax.legend(["Model", "Baseline"])
