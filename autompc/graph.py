# Created by William Edwards (wre2@illinois.edu)

from abc import abstractmethod

class Graph:
    def __init__(self, system, model, configuration, need_training_eval=False):
        self.system = system
        self.model = model
        self.configuration = configuration
        self.need_training_eval = need_training_eval

    @abstractmethod
    def add_traj(self, predictor, traj, training=False):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, fig):
        """
        Draws the corresponding graph.
        """
        raise NotImplementedError
