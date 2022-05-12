# Created by William Edwards, (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class ThresholdCost(Cost):
    def __init__(self, system, goal, threshold, obs_range=None, observations=None):
        """
        Create threshold cost. Returns 1 for every time steps
        where :math:`||x - x_\\textrm{goal}||_\\infty > \\textrm{threshold}`.
        The check is performed only over the observation dimensions from
        obs_range[0] to obs_range[1].

        Parameters
        ----------
        system : System
            Robot system object

        goal : Numpy array
            Goal position

        obs_range : (int, int)
            First (inclusive and last (exclusive) index of observations
            for which goal is specified.  If neither this field nor
            observations is set, default is full observation range.

        observations : [str]
            List of observation names for which goal is specified.
            Supersedes obs_range when present.
        """
        super().__init__(system)
        self._threshold = np.copy(threshold)
        self._obs_idxs = None
        if obs_range is not None:
            self._obs_idxs = list(range(obs_range[0], obs_range[1]))
        if observations is not None:
            self._obs_idxs = [system.observations.index(obs) for obs in observations]
        if self._obs_idxs is None:
            self._obs_idxs = list(range(0, system.obs_dim))
        self.set_goal(goal)

    def set_goal(self, goal):
        if len(goal) < self.system.obs_dim:
            self._goal = np.zeros(self.system.obs_dim)
            self._goal[self._obs_idxs] = goal
        else:
            self._goal = np.copy(goal)

    def incremental(self, obs, ctrl):
        if (la.norm(obs[self._obs_idxs] - self._goal[self._obs_idxs], np.inf) 
                > self._threshold):
            return 1.0
        else:
            return 0.0

    def terminal(self, obs):
        return 0.0


class BoxThresholdCost(Cost):
    def __init__(self, system, limits, goal=None):
        """
        Create Box threshold cost. Returns 1 for every time steps
        where observation is outisde of limits.

        Paramters
        ---------
        system : System
            System cost is computed for

        limits : numpy array of shape (system.obs_dim, 2)
            Upper and lower limits.  Use +np.inf or -np.inf
            to allow certain dimensions unbounded.

        goal : numpy array of size system.obs_dim
            Goal state.  Not used directly for computing cost, but
            may be used by downstream cost factories.
        """
        super().__init__(system)
        self._limits = np.copy(limits)

        if goal is not None:
            self.properties['goal'] = np.copy(goal)

    def incremental(self, obs, ctrl):
        if (obs < self._limits[:,0]).any() or (obs > self._limits[:,1]).any():
            return 1.0
        else:
            return 0.0

    def terminal(self, obs):
        return 0.0
