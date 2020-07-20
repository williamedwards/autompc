# Created by William Edwards (wre2@illinois.edu)

import math
from pdb import set_trace
import numpy as np

from ..evaluator import Evaluator, CachingPredictor
from .. import utils

class HoldoutEvaluator(Evaluator):
    def __init__(self, *args, holdout_prop = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        holdout_size = round(holdout_prop * len(self.trajs))
        holdout_indices = self.rng.choice(np.arange(len(self.trajs)), 
                holdout_size, replace=False)
        self.holdout = [self.trajs[i] for i in sorted(holdout_indices)]
        self.training_set = []
        for traj in self.trajs:
            if traj not in self.holdout:
                self.training_set.append(traj)

    def __call__(self, model, configuration):
        m = utils.make_model(self.system, model, configuration)
        m.train(self.training_set)
        primary_metric_values = np.zeros(len(self.holdout))
        secondray_metric_values = np.zeros((len(self.holdout),
            len(self.secondary_metrics)))
        for i, traj in enumerate(self.holdout):
            predictor = CachingPredictor(traj, m)
            primary_metric_values[i] = self.primary_metric(predictor, traj)
            for j, metric in enumerate(self.secondary_metrics):
                secondary_metric_values[i, j] = metric(predictor, traj)

        primary_metric_value = self.primary_metric.accumulate(primary_metric_values)
        secondary_metric_value = np.array(len(self.secondary_metrics))

        for j, metric in enumerate(self.secondary_metrics):
            secondary_metric_value[j] = metric.accumulate(secondary_metric_values[:,j])

        #print("k = {}, score = {}".format(m.k, primary_metric_value))
        print("CFG:")
        print(configuration)
        print("score = {}".format(primary_metric_value))

        return primary_metric_value, secondary_metric_value
