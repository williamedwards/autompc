# Created by William Edwards (wre2@illinois.edu)

# Standard libary includes
import copy

# External libary includes
import numpy as np

class OCP:
    """
    Defines an optimal control problem to be solved.
    """
    def __init__(self, system):
        """
        Task constructor

        Parameters
        ----------
        sytem : System
            Robot system for which task is defined.
        """
        self.system = system

        # Initialize obs and control bounds
        self._obs_bounds = np.zeros((system.obs_dim, 2))
        for i in range(system.obs_dim):
            self._obs_bounds[i, 0] = -np.inf
            self._obs_bounds[i, 1] = np.inf

        self._ctrl_bounds = np.zeros((system.ctrl_dim, 2))
        for i in range(system.ctrl_dim):
            self._ctrl_bounds[i, 0] = -np.inf
            self._ctrl_bounds[i, 1] = np.inf

    def clone(self):
        return copy.deepcopy(self)

    def set_cost(self, cost):
        """
        Sets the task cost

        Parameters
        ----------
        cost : Cost
            Cost to be set
        """
        self.cost = cost

    def get_cost(self):
        """
        Get the task cost

        Returns 
        -------
        : Cost
            Task cost
        """
        return self.cost

    # Handle bounds
    def set_obs_bound(self, obs_label, lower, upper):
        """
        Set a bound for one dimension of  the observation

        Parameters
        ----------
        obs_label : string
            Name of observation dimension to be bounded

        lower : float
            Lower bound

        upper : float
            Upper bound
        """
        idx = self.system.observations.index(obs_label)
        self._obs_bounds[idx,:] = [lower, upper]

    def set_obs_bounds(self, lowers, uppers):
        """
        Set bound for all observation dimensions

        Parameters
        ----------
        lowers : numpy array of size self.system.obs_dim
            Lower bounds

        uppers : numpy array of size self.system.obs_dim
            Upper bounds
        """
        self._obs_bounds[:,0] = lowers
        self._obs_bounds[:,1] = uppers

    def set_ctrl_bound(self, ctrl_label, lower, upper):
        """
        Set a bound for one dimension of the control

        Parameters
        ----------
        ctrl_label : string
            Name of control dimension to be bounded

        lower : float
            Lower bound

        upper : float
            Upper bound
        """
        idx = self.system.controls.index(ctrl_label)
        self._ctrl_bounds[idx,:] = [lower, upper]

    def set_ctrl_bounds(self, lowers, uppers):
        """
        Set bound for all control dimensions

        Parameters
        ----------
        lowers : numpy array of size self.system.ctrl_dim
            Lower bounds

        uppers : numpy array of size self.system.ctrl_dim
            Upper bounds
        """
        self._ctrl_bounds[:,0] = lowers
        self._ctrl_bounds[:,1] = uppers

    @property
    def are_obs_bounded(self):
        """
        Check whether task has observation bounds

        Returns
        -------
        : bool
            True if any observation dimension is bounded
        """
        for i in range(self.system.obs_dim):
            if (self._obs_bounds[i, 0] != -np.inf 
                    or self._obs_bounds[i, 1] != np.inf):
                return True
        return False

    @property
    def are_ctrl_bounded(self):
        """
        Check whether task has control bounds

        Returns 
        -------
        : bool
            True if any control dimension is bounded
        """
        for i in range(self.system.ctrl_dim):
            if (self._ctrl_bounds[i, 0] != -np.inf 
                    or self._ctrl_bounds[i, 1] != np.inf):
                return True
        return False

    def get_obs_bounds(self):
        """
        Get observation bounds. If unbounded, lower and upper bound
        are -np.inf and +np.inf respectively.
        Returns
        -------
        : numpy array of shape (self.system.obs_dim, 2)
            Observation bounds
        """
        return self._obs_bounds.copy()

    def get_ctrl_bounds(self):
        """
        Get control bounds. If unbounded, lower and upper bound
        are -np.inf and +np.inf respectively.
        Returns
        -------
        : numpy array of shape (self.system.ctrl_dim, 2)
            Control bounds
        """
        return self._ctrl_bounds.copy()

class PrototypeCost:
    def __init__(self):
        pass

class PrototypeOCP:
    """
    PrototypeOCP represents only the compatibility properties of
    an OCP.  This is used for checking compatibility with as a little
    overhead as possible.
    """
    def __init__(self, ocp, cost=None):
        self.are_obs_bounded = ocp.are_obs_bounded
        self.are_ctrl_bounded = ocp.are_ctrl_bounded
        if cost is None:
            cost = ocp.cost
        self.cost = PrototypeCost()
        self.cost.is_quad = cost.is_quad
        self.cost.is_convex = cost.is_convex
        self.cost.is_diff = cost.is_diff
        self.cost.is_twice_diff = cost.is_twice_diff
        self.cost.has_goal = cost.has_goal

    def get_cost(self):
        return self.cost