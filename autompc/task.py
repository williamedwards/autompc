# Created by William Edwards (wre2@illinois.edu)

import copy
import numpy as np

class NumStepsTermCond:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, traj):
        return len(traj) >= self.num_steps

class Task:
    """
    Defines a control task to be solved
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

        self._term_cond = None
        self._num_steps = None
        self._init_obs  = None

    def set_num_steps(self, num_steps):
        """
        Sets maximum number of steps as the task terminiation
        condition.

        Parameters
        ----------
        num_steps : int
            Maximum number of steps.
        """
        self._term_cond = NumStepsTermCond(num_steps)
        self._num_steps = num_steps

    def has_num_steps(self):
        """
        Check whether task has a maximum number of steps for the
        task.

        Returns
        -------
        : bool
            True if maximum number of steps is set
        """
        return self._num_steps is not None

    def get_num_steps(self):
        """
        Returns the maxium number steps if available. None otherwise.
        """
        return self._num_steps

    def term_cond(self, traj):
        """
        Checks the task termination condition.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to check termination condition.

        Returns
        -------
        : bool
            True if termination condition met.
        """
        if self._term_cond is not None:
            return self._term_cond(traj)
        else:
            return False

    def set_term_cond(self, term_cond):
        """
        Set the task termination condition

        Parameters
        ----------
        term_cond : Function, Trajectory -> bool
            Termination condition function.
        """
        self._term_cond = term_cond

    def set_ocp(self, ocp):
        """
        Sets the task ocp

        Parameters
        ----------
        ocp : OCP
            Control problem to be solved
        """
        self.ocp = ocp

    def get_ocp(self):
        """
        Get the task cost

        Returns 
        -------
        : OCP
            Task ocp
        """
        return self.ocp

    def set_init_obs(self, init_obs):
        """
        Sets the initial observation for the task.

        Parameters
        ----------
        init_obs : numpy array of size self.system.obs_dim
            Initial observation
        """
        init_obs = np.array(init_obs)
        if not init_obs.shape == (self.system.obs_dim,):
            raise ValueError("init_obs has wrong shape")
        self._init_obs = init_obs

    def get_init_obs(self):
        """
        Get the initial observation for the task

        Returns : numpy array of size self.system.obs_dim
            Initial observation
        """
        return self._init_obs

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
