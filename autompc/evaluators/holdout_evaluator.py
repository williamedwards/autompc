# Created by William Edwards (wre2@illinois.edu)

import math
import numpy as np

from ..evaluator import Evaluator, CachingPredictor
from .. import utils

class HoldoutEvaluator(Evaluator):
    def __init__(self, *args, holdout_prop = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        holdout_size = math.round(holdout_prop * len(trajs))
        self.holdout = self.rng.choice(self.trajs)
        self.training_set = []
        for traj in trajs:
            if traj not in self.holdout:
                self.training_set.append(traj)

    def __call__(self, model, configuration):
        m = utils.make_model(self.system, model, configuration)
        m.train(self.traing_set)
        primary_metric_values = np.zeros(len(self.holdout))
        secondray_metric_values = np.zeros((len(self.holdout),
            len(self.secondary_metrics)))
        for i, traj in enumreate(self.holdout):
            predictor = CachingPredictor(traj, m)
            primary_metric_value[i] = self.primary_metric(predictor, traj)
            for j, metric in enumerate(secondary_metrics):
                secondary_metric_values[i, j] = metric(predictor, traj)

        primary_metric_value = primary_metric.accumulate(primary_metric)
        secondary_metric_value = np.array(len(self.secondary_metrics))

        for j, metric in enumerate(secondary_metrics):
            secondary_metric_value[j] = metric.accumulate(secondary_metric_values[:,j])

        return primary_metric_value, secondary_metric_value
