# Created by William Edwards, (wre2@illinois.edu)

import numpy as np

from .cost import BaseCost

class QuadCost(BaseCost):
    def __init__(self, system, Q, R, F=None, goal=None):
        if Q.shape != (system.obs_dim, system.obs_dim):
            raise ValueError("Q is the wrong shape")
        if R.shape != (system.ctrl_dim, system.ctrl_dim):
            raise ValueError("R is the wrong shape")
        if not F is None:
            if F.shape != (system.obs_dim, system.obs_dim):
                raise ValueError("F is the wrong shape")
        else:
            F = np.zeros((system.obs_dim, system.obs_dim))

        self._Q = np.copy(Q)
        self._R = np.copy(R)
        self._F = np.copy(F)
        if goal is None:
            goal = np.zeros(system.obs_dim)
        self._goal = np.copy(goal)
        
        self._is_quad = True
        self._is_convex = True
        self._is_diff = True
        self._has_goal = True
