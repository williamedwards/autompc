# Created by William Edwards, (wre2@illinois.edu)

import numpy as np

from abc import ABC, abstractmethod

class Cost(ABC):
    """
    Base class for cost functions.
    """
    def __init__(self, system):
        """
        Create cost

        Parameters
        ----------
        system : System
            Robot system for which cost will be evaluated
        """
        self.system = system
        self._is_quad = False
        self._is_convex = False
        self._is_diff = False
        self._is_twice_diff = False
        self._has_goal = False

    def __call__(self, traj):
        """
        Evaluate cost on whole trajectory

        Parameters
        ----------
        traj : Trajectory
            Trajectory to evaluate
        """
        cost = 0.0
        for i in range(len(traj)):
            cost += self.eval_obs_cost(traj[i].obs)
            cost += self.eval_ctrl_cost(traj[i].ctrl)
        cost += self.eval_term_obs_cost(traj[-1].obs)
        return cost

    def get_cost_matrices(self):
        """
        Return quadratic Q, R, and F matrices. Raises exception
        for non-quadratic cost.
        """
        if self.is_quad:
            return np.copy(self._Q), np.copy(self._R), np.copy(self._F)
        else:
            raise ValueError("Cost is not quadratic.")

    def get_goal(self):
        """
        Returns the cost goal state if available. Raises exception
        if cost does not have goal.

        Returns : numpy array of size self.system.obs_dim
            Goal state
        """
        if self.has_goal:
            return np.copy(self._goal)
        else:
            raise ValueError("Cost does not have goal")

    def eval_obs_cost(self, obs):
        """
        Evaluates observation cost at a particular time step.
        Raises exception if not implemented.

        Parameters
        ----------
        obs : self.system.obs_dim
            Observation

        Returns : float
            Cost
        """
        if self.is_quad:
            obst = obs - self._goal
            return obst.T @ self._Q @ obst
        else:
            raise NotImplementedError

    def eval_obs_cost_diff(self, obs):
        """
        Evaluates the observation cost at a particular time
        steps and computes Jacobian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim)
            Cost, Jacobian
        """
        if self.is_quad:
            obst = obs - self._goal
            return obst.T @ self._Q @ obst, (self._Q + self._Q.T) @ obst
        else:
            raise NotImplementedError

    def eval_obs_cost_hess(self, obs):
        """
        Evaluates the observation cost at a particular time
        steps and computes Jacobian and Hessian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim,
                  numpy array of shape (self.system.obs_dim, self.system.obsd_im))
            Cost, Jacobian, Hessian
        """
        if self.is_quad:
            obst = obs - self._goal
            return (obst.T @ self._Q @ obst, 
                    (self._Q + self._Q.T) @ obst,
                    self._Q + self._Q.T)
        else:
            raise NotImplementedError

    def eval_ctrl_cost(self, ctrl):
        """
        Evaluates control cost at a particular time step.
        Raises exception if not implemented.

        Parameters
        ----------
        obs : self.system.ctrl_dim
            Control

        Returns : float
            Cost
        """
        if self.is_quad:
            return ctrl.T @ self._R @ ctrl
        else:
            raise NotImplementedError

    def eval_ctrl_cost_diff(self, ctrl):
        """
        Evaluates the control cost at a particular time
        step and computes Jacobian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.ctrl_dim)
            Cost, Jacobian
        """
        if self.is_quad:
            return ctrl.T @ self._R @ ctrl, (self._R + self._R.T) @ ctrl
        else:
            raise NotImplementedError

    def eval_ctrl_cost_hess(self, ctrl):
        """
        Evaluates the control cost at a particular time
        steps and computes Jacobian and Hessian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.ctrl_dim, numpy array of shape (self.system.ctrl_dim, self.system.ctrl_dim))
            Cost, Jacobian, Hessian
        """
        if self.is_quad:
            return (ctrl.T @ self._R @ ctrl, 
                    (self._R + self._R.T) @ ctrl,
                    self._R + self._R.T)
        else:
            raise NotImplementedError

    def eval_term_obs_cost(self, obs):
        """
        Evaluates terminal observation cost.
        Raises exception if not implemented.

        Parameters
        ----------
        obs : self.system.obs_dim
            Observation

        Returns : float
            Cost
        """
        if self.is_quad:
            obst = obs - self._goal
            return obst.T @ self._F @ obst
        else:
            raise NotImplementedError

    def eval_term_obs_cost_diff(self, obs):
        """
        Evaluates the terminal observation cost
        and computes Jacobian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim)
            Cost, Jacobian
        """
        if self.is_quad:
            return obs.T @ self._F @ obs, (self._F + self._F.T) @ obs
        else:
            raise NotImplementedError

    def eval_term_obs_cost_hess(self, obs):
        """
        Evaluates the terminal observation cost
        and computes Jacobian and Hessian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim, numpy array of shape (self.system.obs_dim, self.system.obsd_im))
            Cost, Jacobian, Hessian
        """
        if self.is_quad:
            return (obs.T @ self._F @ obs, 
                    (self._F + self._F.T) @ obs,
                    self._F + self._F.T)
        else:
            raise NotImplementedError

    @property
    def is_quad(self):
        """
        True if cost is quadratic.
        """
        return self._is_quad

    @property
    def is_convex(self):
        """
        True if cost is convex.
        """
        return self._is_convex

    @property
    def is_diff(self):
        """
        True if cost is differentiable.
        """
        return self._is_diff

    @property
    def is_twice_diff(self):
        """
        True if cost is twice differentiable
        """
        return self._is_twice_diff

    @property
    def has_goal(self):
        """
        True if cost has goal
        """
        return self._has_goal

    def __add__(self, other):
        from .sum_cost import SumCost
        if isinstance(other, SumCost):
            return other.__radd__(self)
        else:
            return SumCost(self.system, [self, other])
