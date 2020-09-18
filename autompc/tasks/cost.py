# Created by William Edwards, (wre2@illinois.edu)

import numpy as np

from abc import ABC, abstractmethod

class BaseCost(ABC):
    def __init__(self, system):
        self.system = system
        self._is_quad = False
        self._is_convex = False
        self._is_diff = False

    def __call__(self, traj):
        cost = 0.0
        for i in range(len(traj)):
            cost += self.eval_obs_cost(traj[i].obs)
            cost += self.eval_ctrl_cost(traj[i].ctrl)
        cost += self.eval_term_obs_cost(traj[-1].obs)
        return cost

    def get_cost_matrices(self):
        """
        Return quadratic cost matrices. Q,
        R, and F.
        """
        if self.is_quad:
            return np.copy(self._Q), np.copy(self._R), np.copy(self._F)
        else:
            raise ValueError("Cost is not quadratic.")

    def eval_obs_cost(self, obs):
        """
        Returns additive observation cost of the form
        obs -> float.
        """
        if self.is_quad:
            return obs.T @ self._Q @ obs
        else:
            raise NotImplementedError

    def eval_obs_cost_diff(self, obs):
        """
        Returns additive observation cost of the form
        obs -> float, jac
        """
        if self.is_quad:
            return obs.T @ self._Q @ obs, (self._Q + self._Q.T) @ obs
        else:
            raise NotImplementedError

    def eval_ctrl_cost(self, ctrl):
        """
        Returns additive observation cost of the form
        ctrl -> float.
        """
        if self.is_quad:
            return ctrl.T @ self._R @ ctrl
        else:
            raise NotImplementedError

    def eval_ctrl_cost_diff(self, ctrl):
        """
        Returns additive observation cost of the form
        ctrl -> float, jac
        """
        if self.is_quad:
            return ctrl.T @ self._R @ ctrl, (self._R + self._R.T) @ ctrl
        else:
            raise NotImplementedError

    def eval_term_obs_cost(self, obs):
        """
        Returns additive observation cost of the form
        obs -> float.
        """
        if self.is_quad:
            return obs.T @ self._F @ obs
        else:
            raise NotImplementedError

    def eval_term_obs_cost_diff(self, obs):
        """
        Returns additive observation cost of the form
        obs -> float, jac
        """
        if self.is_quad:
            return obs.T @ self._F @ obs, (self._F + self._F.T) @ obs
        else:
            raise NotImplementedError

    @property
    def is_quad(self):
        return self._is_quad

    @property
    def is_convex(self):
        return self._is_convex

    @property
    def is_diff(self):
        return self._is_diff

    def is_diff(self):
        return self._is_diff
