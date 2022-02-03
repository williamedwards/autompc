# Created by William Edwards, (wre2@illinois.edu)

import numpy as np

from .cost import Cost

class QuadCost(Cost):
    def __init__(self, system, Q, R, F=None, goal=None):
        """
        Create quadratic cost.

        Parameters
        ----------
        system : System
            System for cost

        Q : numpy array of shape (self.obs_dim, self.obs_dim)
            Observation cost matrix

        R : numpy array of shape (self.ctrl_dim, self.ctrl_dim)
            Control cost matrix

        F : numpy array of shape (self.ctrl_dim, self.ctrl_dim)
            Terminal observation cost matrix

        goal : numpy array of shape self.obs_dim
            Goal state. Default is zero state
        """
        super().__init__(system)
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
        self._is_twice_diff = True
        self._has_goal = True

    is_quad = True
    is_convex = True
    is_diff = True
    is_twice_diff = True
    has_goal = True
