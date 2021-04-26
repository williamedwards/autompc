# Created by William Edwards (wre2@illinois.edu)

import math, time
from pdb import set_trace
import numpy as np

from .evaluator import Evaluator
from .. import utils

class HoldoutModelEvaluator(Evaluator):
    def __init__(self, *args, holdout_prop = 0.1, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        holdout_size = round(holdout_prop * len(self.trajs))
        holdout_indices = self.rng.choice(np.arange(len(self.trajs)), 
                holdout_size, replace=False)
        self.holdout = [self.trajs[i] for i in sorted(holdout_indices)]
        self.training_set = []
        for traj in self.trajs:
            if traj not in self.holdout:
                self.training_set.append(traj)

    def __call__(self, model_factory, configuration):
        if self.verbose:
            print("Evaluating Configuration:")
            print(configuration)
            print("----")
        m = model_factory(configuration, self.training_set)

        metric_value = self.metric(m, self.holdout)

        return metric_value
