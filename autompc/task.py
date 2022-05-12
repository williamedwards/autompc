# Created by William Edwards (wre2@illinois.edu)

import copy
import numpy as np
from .system import System
from .ocp import OCP
from .costs import Cost

class NumStepsTermCond:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, traj):
        return len(traj) >= self.num_steps

class Task(OCP):
    """
    Defines a finite-horizon control task for the tuner.  This generalizes
    an OCP in that 1) it specifies a start state, 2) a trial is terminated
    at a fixed horizon or when a termination criterion is reached.
    A Task is something evaluatable, whereas an OCP simply specifies the
    constraints and preferences for the system behavior.
    """
    def __init__(self, system : System, cost : Cost = None):
        """
        Task constructor

        Parameters
        ----------
        system : System
            Robot system for which task is defined.
        cost : Cost
            The objective function.
        """
        super().__init__(system,cost)
        
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
        term_cond : Function(Trajectory) -> bool
            Termination condition function.
        """
        self._term_cond = term_cond

    def set_goal(self, goal) -> None:
        """
        Set the task's goal state.

        Parameters
        ----------
        goal: numpy array of size system.obs_dim.
        """
        self.get_cost().goal = goal
    
    def get_goal(self):
        """
        Retrieves the task's goal state.
        """
        return self.get_cost().goal

    def set_ocp(self, ocp : OCP) -> None:
        """
        Sets the task ocp

        Parameters
        ----------
        ocp : OCP
            Control problem to be solved
        """
        self.set_cost(ocp.get_cost())
        self.set_ctrl_bounds(ocp.get_ctrl_bounds())
        self.set_obs_bounds(ocp.get_obs_bounds())

    def get_ocp(self) -> OCP:
        """
        Get the task ocp

        Returns 
        -------
        : OCP
            Task ocp
        """
        res = OCP(self.system,self.cost)
        res.set_obs_bounds(self.get_obs_bounds())
        res.set_ctrl_bounds(self.get_ctrl_bounds())
        return res

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
