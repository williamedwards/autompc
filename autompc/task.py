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
