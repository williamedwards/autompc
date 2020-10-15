# Created by William Edwards, (wre2@illinois.edu)

import numpy as np

from .cost import BaseCost

class QuadCost(BaseCost):
    def __init__(self, system, Q, R, F=None, x0=None):
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
        if x0 is None:
            x0 = np.zeros(system.obs_dim)
        self._x0 = np.copy(x0)
        
        self._is_quad = True
        self._is_convex = True
        self._is_diff = True
