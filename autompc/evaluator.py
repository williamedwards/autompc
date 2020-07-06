# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod

class CachingPredictor:
    def __init__(self, traj, model):
        self.traj = traj
        self.model = model
        self.cache = dict()

    def __call__(self, start, horizon):
        state = self._get_state(start, horizon)
        return state[:self.model.system.obs_dim]

    def _get_state(self, start, horizon):
        if (start, horizon) in self.cache:
            return self.cache[(start,horizon)]
        if horizon == 1:
            curr_state = self.model.traj_to_state(self.traj[:start+1])
        else:
            curr_state = self._get_state(start, horizon-1)
        #print("start={}, horizon={}, curr_state={}".format(start, horizon, curr_state))
        return self.model.pred(curr_state, self.traj[start+horizon-1].ctrl)

class Evaluator(ABC):
    def __init__(self, system, trajs, primary_metric, rng):
        self.system = system
        self.trajs = trajs
        self.primary_metric = primary_metric
        self.secondary_metrics = []
        self.graphers = []
        self.rng = rng

    @abstractmethod
    def __call__(self, model, configuration):
        """
        Accepts the model class and the configuration.
        Returns
        """
        raise NotImplementedError

    def add_secondary_metric(self, metric):
        self.secondary_metrics.append(metric)

    def add_grapher(self, grapher):
        self.graphers = []
