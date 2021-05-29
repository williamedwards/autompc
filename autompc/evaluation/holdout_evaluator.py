# Created by William Edwards (wre2@illinois.edu)

import math, time
from pdb import set_trace
import numpy as np

from .evaluator import ModelEvaluator
from .. import utils

class HoldoutModelEvaluator(ModelEvaluator):
    """
    Evaluate model prediction accuracy according to a holdout set.
    """
    def __init__(self, *args, holdout_prop = 0.1, holdout_set=None, verbose=False, **kwargs):
        """
        Parameters
        ----------
        system : System
            System for which prediction accuracy is evaluated
        trajs : List of Trajectory
            Trajectories to be used for evaluation
        metric : string or function (model, [Trajectory] -> float)
            Metric which evaluates the model against a set of trajectories.
            If string, one of "rmse", "rmsmens". See `model_metrics` for
            more details.
        rng : np.random.Generator
            Random number generator used in evaluation 
        horizon : int
            Prediction horizon used in certain metrics. Default is 1.
        holdout_prop : float
            Proportion of dataset to hold out for evaluation
        holdout_set : List of Trajectory
            This argument can be passed to explicitly set holdout set, rather
            than randomly selecting it. Pass None otherwise.
        verbose : bool
            Whether to print information during evaluation
        """
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        if holdout_set is None:
            holdout_size = round(holdout_prop * len(self.trajs))
            holdout_indices = self.rng.choice(np.arange(len(self.trajs)), 
                    holdout_size, replace=False)
            self.holdout = [self.trajs[i] for i in sorted(holdout_indices)]
        else:
            self.holdout = holdout_set
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
