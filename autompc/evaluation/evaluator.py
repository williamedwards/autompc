# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from collections import defaultdict
from .model_metrics import get_model_rmse,get_model_rmsmens

class ModelEvaluator(ABC):
    """
    The ModelEvaluator class evaluates models by prediction accuracy.
    """
    def __init__(self, system, trajs, metric, rng, horizon=1):
        """s
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
        """
        self.system = system
        self.trajs = trajs
        self.rng = rng
        if isinstance(metric, str):
            if metric == "rmse":
                self.metric = lambda model, trajs: get_model_rmse(model, 
                        trajs, horizon=horizon)
            elif metric == "rmsmens":
                self.metric = lambda model, trajs: get_model_rmsmens(model, 
                        trajs, horizon=horizon)
        else:
            self.metric = metric

    @abstractmethod
    def __call__(self, model, configuration):
        """
        Accepts the model class and the configuration.

        Parameters
        ----------
        model : Model
            Model factory evaluate

        configuration : Configuration
            Hyperparameter configuration used to
            train model

        Returns
        -------
        score : float
            Evaluated score.
        """
        raise NotImplementedError
