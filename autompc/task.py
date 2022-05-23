# Created by William Edwards (wre2@illinois.edu)

import copy
import weakref
from typing import Tuple,Union
import numpy as np
from .system import System
from .ocp import OCP
from .costs import Cost
from .dynamics import Dynamics
from .policy import Policy
from .trajectory import Trajectory


class NumStepsTermCond:
    """Terminates after num_steps steps"""
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, traj):
        return len(traj) >= self.num_steps

class ControlInfeasibleTermCond:
    """Terminates when the control becomes infeasible"""
    def __init__(self, ocp : OCP):
        self.ocp = ocp

    def __call__(self, traj):
        return not self.ocp.is_ctrl_feasible(traj.ctrls[-1])

class StateInfeasibleTermCond:
    """Terminates when the state becomes infeasible"""
    def __init__(self, ocp : OCP):
        self.ocp = ocp

    def __call__(self, traj):
        return not self.ocp.is_obs_feasible(traj.obs[-1])

class GoalRadiusTermCond:
    """Terminates when ||x-goal|| <= radius"""
    def __init__(self, goal, radius):
        self.goal = goal
        self.radius = radius

    def __call__(self, traj):
        return np.linalg.norm(traj.obs[-1] - self.goal) <= self.radius

class GoalThresholdTermCond:
    """Terminates when |x-goal| <= distances, elementwise"""
    def __init__(self, goal, distances):
        self.goal = goal
        self.distances = distances

    def __call__(self, traj):
        return (np.abs(traj.obs[-1] - self.goal) <= self.distances).all()


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
        
        self._term_conds = dict()
        self._num_steps = None
        self._init_obs  = None

    def set_num_steps(self, num_steps : int) -> None:
        """
        Sets maximum number of steps as the task terminiation
        condition.

        Parameters
        ----------
        num_steps : int
            Maximum number of steps.
        """
        self._term_conds['max_steps'] = NumStepsTermCond(num_steps)
        self._num_steps = num_steps

    def has_num_steps(self) -> bool:
        """
        Check whether task has a maximum number of steps for the
        task.

        Returns
        -------
        : bool
            True if maximum number of steps is set
        """
        return self._num_steps is not None

    def get_num_steps(self) -> int:
        """
        Returns the maxium number steps if available. None otherwise.
        """
        return self._num_steps

    def term_cond(self, traj : Trajectory) -> Union[str,None]:
        """
        Checks the task termination condition.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to check termination condition.

        Returns
        -------
        : str or None
            True if termination condition met.
        """
        for name,term_cond in self._term_conds.items():
            if term_cond(traj):
                return name
        return None

    def add_term_cond(self, name, term_cond) -> None:
        """
        Adds a task termination condition

        Parameters
        ----------
        name : str
            The name of the termination condition.
        term_cond : Function(Trajectory) -> bool
            Termination condition function.
        """
        self._term_conds[name] = term_cond

    def set_goal(self, goal) -> None:
        """
        Set the task's goal state.

        Parameters
        ----------
        goal: numpy array of size system.obs_dim.
        """
        self.get_cost().goal = goal
    
    def get_goal(self) -> np.ndarray:
        """
        Retrieves the task's goal state.
        """
        return self.get_cost().goal

    def set_goal_term(self, radius_or_thresholds : Union[float,np.ndarray]) -> None:
        """Tells the task to terminate when the state reaches a certain
        radius of the goal state, or thresholds.
        """
        if hasattr(radius_or_thresholds,'__iter__'):
            self.add_term_cond(['goal_reached'],GoalThresholdTermCond(self.get_goal(),radius_or_thresholds))
        else:
            self.add_term_cond(['goal_reached'],GoalRadiusTermCond(self.get_goal(),radius_or_thresholds))

    def set_ocp(self, ocp : OCP) -> None:
        """
        Sets the task ocp

        Parameters
        ----------
        ocp : OCP
            Costs and constraints for the control problem
        """
        self.set_cost(ocp.get_cost())
        self.set_ctrl_bounds(ocp.get_ctrl_bounds())
        self.set_obs_bounds(ocp.get_obs_bounds())

    def get_ocp(self) -> OCP:
        """
        Get the task ocp (costs and constraints)

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

    def simulate(self, policy : Policy, dynamics : Dynamics) -> Tuple[Trajectory,float,Union[str,None]]:
        """Simulates a controller on this task given some dynamics model.

        Note that if the policy is a controller, it is not reset at the start --
        this will need to be done manually.

        Returns : (trajectory,cost,term_cond)
            The rolled out trajectory, the total OCP cost, and the termination
            condition.  Termination condition can be None (max iters reached),
            'max_steps', 'control_infeasible', 'state_infeasible', or
            'goal_reached', in addition to any user-defined conditions.
        """
        from .utils.simulation import simulate
        if self.has_num_steps():
            traj = simulate(policy, self.get_init_obs(),
                dynamics, self.term_cond, 
                max_steps=self.get_num_steps())
        else:
            traj = simulate(policy, self.get_init_obs(),
                dynamics, term_cond=self.term_cond)
        cost = self.get_ocp().get_cost()
        rollout_cost = cost(traj)
        return traj,rollout_cost,self.term_cond(traj)

