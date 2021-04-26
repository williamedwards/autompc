# Created by William Edwards, (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import BaseCost

class ThresholdCost(BaseCost):
    def __init__(self, system, goal, obs_range, threshold):
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
