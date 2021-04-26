# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from collections import defaultdict

class Evaluator(ABC):
    def __init__(self, system, trajs, metric, rng):
        self.system = system
        self.trajs = trajs
        self.metric = metric
        self.rng = rng

    @abstractmethod
    def __call__(self, model, configuration):
        """
        Accepts the model class and the configuration.
        Returns
        """
        raise NotImplementedError
