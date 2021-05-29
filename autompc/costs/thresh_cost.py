# Created by William Edwards, (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class ThresholdCost(Cost):
    def __init__(self, system, goal, obs_range, threshold):
        """
        Create threshold cost. Returns 1 for every time steps
        where :math:`||x - x_\\textrm{goal}||_\\infty > \\textrm{threshold}`.
        The check is performed only over the observation dimensions from
        obs_range[0] to obs_range[1].
        """
        super().__init__(system)
        self._goal = np.copy(goal)
        self._threshold = np.copy(threshold)
        self._obs_range = obs_range[:]

        self._is_quad = False
        self._is_convex = False
        self._is_diff = False
        self._is_twice_diff = False
        self._has_goal = True

    def eval_obs_cost(self, obs):
        if (la.norm(obs[self._obs_range[0]:self._obs_range[1]] - self._goal[self._obs_range[0]:self._obs_range[1]], np.inf) 
                > self._threshold):
            return 1.0
        else:
            return 0.0

    def eval_ctrl_cost(self, ctrl):
        return 0.0

    def eval_term_obs_cost(self, obs):
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

        self._is_quad = False
        self._is_convex = False
        self._is_diff = False
        self._is_twice_diff = False

        if goal is None:
            self._has_goal = False
        else:
            self._goal = np.copy(goal)
            self._has_goal = True

    def eval_obs_cost(self, obs):
        if (obs < self._limits[:,0]).any() or (obs > self._limits[:,1]).any():
            return 1.0
        else:
            return 0.0

    def eval_ctrl_cost(self, ctrl):
        return 0.0

    def eval_term_obs_cost(self, obs):
        return 0.0
