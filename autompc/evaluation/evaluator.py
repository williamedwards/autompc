# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from collections import defaultdict
from .model_metrics import get_model_rmse

class Evaluator(ABC):
    def __init__(self, system, trajs, metric, rng, horizon=1):
        self.system = system
        self.trajs = trajs
        self.rng = rng
        if isinstance(metric, str):
            if metric == "rmse":
                self.metric = lambda model, trajs: get_model_rmse(model, 
                        trajs, horizon=horizon)
        else:
            self.metric = metric

    @abstractmethod
    def __call__(self, model, configuration):
        """
        Accepts the model class and the configuration.
        Returns
        """
        raise NotImplementedError
