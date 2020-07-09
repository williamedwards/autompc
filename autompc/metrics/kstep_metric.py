# Created by William Edwards (wre2@illinois.edu)

import numpy.linalg as la
import numpy as np
from pdb import set_trace

from ..metric import Metric

class RmseKstepMetric(Metric):
    def __init__(self, system, k=1):
        super().__init__(system)
        self.k = k

    def __call__(self, predictor, traj):
        sumsqe = 0.0
        for i in range(len(traj)-self.k):
            pred_obs = predictor(i, self.k)
            actual_obs = traj[i+self.k].obs
            sumsqe += la.norm(pred_obs - actual_obs)**2
        return sumsqe / len(traj)

    def accumulate(self, values):
        return np.sqrt(np.mean(values))
