# Created by William Edwards (wre2@illinois.edu)

from abc import abstractmethod

class Graph:
    def __init__(self, system, model, configuration):
        self.system = system
        self.model = model
        self.configuration = configuration

    @abstractmethod
    def add_traj(self, predictor, traj):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, ax):
        """
        Draws the corresponding graph.
        """
        raise NotImplementedError
