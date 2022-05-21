# Created by William Edwards (wre2@illinois.edu)

# Standard libary includes
import copy

# External libary includes
import numpy as np

# Internal module includes
from ..system import System
from ..costs.cost import Cost
from ..trajectory import Trajectory

class OCP:
    """
    Defines an optimal control problem to be solved.
    """
    def __init__(self, system : System, cost : Cost = None):
        """
        Parameters
        ----------
        system : System
            Dynamical system for which the task is defined.
        cost : Cost
            Cost specifying trajectory preferences (lower is better).
        """
        self.system = system
        self.cost = cost

        # Initialize obs and control bounds
        self._obs_bounds = np.zeros((system.obs_dim, 2))
        self._obs_bounds[:,0] = -np.inf
        self._obs_bounds[:,1] = np.inf

        self._ctrl_bounds = np.zeros((system.ctrl_dim, 2))
        self._ctrl_bounds[:, 0] = -np.inf
        self._ctrl_bounds[:, 1] = np.inf

    def clone(self) -> 'OCP':
        return copy.deepcopy(self)

    def set_cost(self, cost : Cost) -> None:
        """
        Sets the task cost

        Parameters
        ----------
        cost : Cost
            Cost to be set
        """
        self.cost = cost

    def get_cost(self) -> Cost:
        """
        Get the task cost

        Returns 
        -------
        : Cost
            Task cost
        """
        return self.cost

    # Handle bounds
    def set_obs_bound(self, obs_label : str, lower : float, upper :float) -> None:
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

    def set_obs_bounds(self, lowers : np.ndarray, uppers : np.ndarray = None) -> None:
        """
        Set bound for all observation dimensions

        Parameters
        ----------
        lowers : numpy array of size self.system.obs_dim
            Lower bounds

        uppers : numpy array of size self.system.obs_dim
            Upper bounds, or if not given, lowers is assumed to be a N*2 array
            containing lower and upper bounds
        """
        if uppers is None:
            if lowers.shape != self._obs_bounds.shape:
                raise ValueError("Invalid size of obs bounds")
            self._obs_bounds = lowers
        else:
            self._obs_bounds[:,0] = lowers
            self._obs_bounds[:,1] = uppers

    def set_ctrl_bound(self, ctrl_label : str, lower : np.ndarray, upper : np.ndarray) -> None:
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

    def set_ctrl_bounds(self, lowers : np.ndarray, uppers : np.ndarray = None) -> None:
        """
        Set bound for all control dimensions

        Parameters
        ----------
        lowers : numpy array of size self.system.ctrl_dim
            Lower bounds

        uppers : numpy array of size self.system.ctrl_dim
            Upper bounds, or if not given, lowers is assumed to be a M*2 array
            containing lower and upper bounds
        """
        if uppers is None:
            if lowers.shape != self._ctrl_bounds.shape:
                raise ValueError("Invalid size of control bounds")
            self._ctrl_bounds = lowers
        else:
            self._ctrl_bounds[:,0] = lowers
            self._ctrl_bounds[:,1] = uppers

    @property
    def are_obs_bounded(self) -> bool:
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
    def are_ctrl_bounded(self) -> bool:
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

    def get_obs_bounds(self) -> np.ndarray:
        """
        Get observation bounds. If unbounded, lower and upper bound
        are -np.inf and +np.inf respectively.
        Returns
        -------
        : numpy array of shape (self.system.obs_dim, 2)
            Observation bounds
        """
        return self._obs_bounds.copy()

    def get_ctrl_bounds(self) -> np.ndarray:
        """
        Get control bounds. If unbounded, lower and upper bound
        are -np.inf and +np.inf respectively.
        Returns
        -------
        : numpy array of shape (self.system.ctrl_dim, 2)
            Control bounds
        """
        return self._ctrl_bounds.copy()
    
    def is_obs_feasible(self, obs : np.ndarray) -> bool:
        """Returns True if the observation is feasible"""
        return not np.any(obs < self._obs_bounds[:,0] | obs > self._obs_bounds[:,1])
    
    def is_ctrl_feasible(self, ctrl : np.ndarray) -> bool:
        """Returns True if the control is feasible"""
        return not np.any(ctrl < self._ctrl_bounds[:,0] | ctrl > self._ctrl_bounds[:,1])

    def is_feasible(self, traj : Trajectory) -> bool:
        """Returns True if the trajectory is feasible"""
        for i in range(len(traj.obs)):
            if np.any(traj.obs[i] < self._obs_bounds[:,0] | traj.obs[i] > self._obs_bounds[:,1]): return False
        for i in range(len(traj.ctrls)):
            if np.any(traj.ctrls[i] < self._ctrl_bounds[:,0] | traj.ctrls[i] > self._ctrl_bounds[:,1]): return False
        return True

    def project_obs(self, obs : np.ndarray) -> np.ndarray:
        """Returns an observation projected into the observation bounds.
        """
        return np.clip(obs,self._obs_bounds[:,0],self._obs_bounds[:,1])

    def project_control(self, ctrl : np.ndarray) -> np.ndarray:
        """Returns a control projected into the control bounds.
        """
        return np.clip(ctrl,self._ctrl_bounds[:,0],self._ctrl_bounds[:,1])

    def project_controls(self, traj : Trajectory) -> Trajectory:
        """Returns a trajectory with all controls feasible.
        Note: does not do anything with the states.
        """
        newctrls = []
        for i in range(len(traj.ctrls)):
            newctrls.append(np.clip(traj.ctrls[i],self._ctrl_bounds[:,0],self._ctrl_bounds[:,1]))
        return Trajectory(traj.system,traj.obs,np.array(newctrls))
