# Created by William Edwards (wre2@illinois.edu)

import numpy as np
from collections import namedtuple
from .system import System
import copy


TimeStep = namedtuple("TimeStep", "obs ctrl")
"""
TimeStep represents a particular time step of a trajectory
and is returned by indexing traj[i].

.. py:attribute:: obs
    Observation. Numpy array of size system.obs_dim

.. py:attribute:: ctrl
    Control. Numpy array of size system.ctrl_dim
"""

class Trajectory:
    """
    The Trajectory object represents a discrete-time state and control
    trajectory.

    The time steps are interpreted such that obs[i] is the state (more
    precisely, the observation) at the i'th time step and ctrls[i] is
    the control executed at that time step.

    You can also access a TimeStep object using the [i] operator.
    """
    def __init__(self, system : System, obs : np.ndarray, ctrls : np.ndarray):
        """
        Parameters
        ----------
        system : System
            The corresponding robot system

        obs : numpy array of shape (size, system.obs_dim)
            Observations at all timesteps

        ctrls : numpy array of shape (size, system.ctrl_dim)
            Controls at all timesteps.
        """
        self._system = system

        # Check inputs
        if len(obs.shape) != 2:
            raise ValueError("Need 2D array of observations")
        if len(ctrls.shape) != 2:
            raise ValueError("Need 2D array of controls")
        if obs.shape[0] != ctrls.shape[0]:
            raise ValueError("obs and ctrls do not have same number of steps, obs has {}, ctrls has {}".format(obs.shape[0],ctrls.shape[0]))
        if obs.shape[1] != system.obs_dim:
            raise ValueError("obs is wrong shape, should be {}, got {}".format((obs.shape[0],system.obs_dim),obs.shape))
        if ctrls.shape[1] != system.ctrl_dim:
            raise ValueError("ctrls is wrong shape, should be {}, got {}".format((obs.shape[1],system.ctrl_dim),ctrls.shape))

        self._obs = obs
        self._ctrls = ctrls
    
    @staticmethod
    def zeros(system : System, size : int) -> 'Trajectory':
        """
        Create an all zeros trajectory.

        Parameters
        ----------
        system : System
            System for trajectory

        size : int
            Size of trajectory
        """
        obs = np.zeros((size, system.obs_dim))
        ctrls = np.zeros((size, system.ctrl_dim))
        return Trajectory(system, obs, ctrls)

    @staticmethod
    def empty(system : System, size : int) -> 'Trajectory':
        """
        Create a trajectory with uninitialized states
        and controls. If not initialized, states/controls
        will be non-deterministic.

        Parameters
        ----------
        system : System
            System for trajectory

        size : int
            Size of trajectory
        """
        obs = np.empty((size, system.obs_dim))
        ctrls = np.empty((size, system.ctrl_dim))
        return Trajectory(system, obs, ctrls)

    def __eq__(self, other):
        return (self._system == other.system
                and np.array_equal(self._obs, other._obs)
                and np.array_equal(self._ctrls, other._ctrls))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if (not isinstance(idx[0], slice) and (idx[0] < -len(self) 
                    or idx[0] >= len(self))):
                raise IndexError("Time index out of range.")
            if idx[1] in self._system.observations:
                obs_idx = self._system.observations.index(idx[1])
                return self._obs[idx[0], obs_idx]
            elif idx[1] in self._system.controls:
                ctrl_idx = self._system.controls.index(idx[1])
                return self._ctrls[idx[0], ctrl_idx]
            else:
                raise IndexError("Unknown label")
        elif isinstance(idx, slice):
            #if idx.start < -len(self) or idx.stop >= len(self):
            #    raise IndexError("Time index out of range.")
            obs = self._obs[idx, :]
            ctrls = self._ctrls[idx, :]
            return Trajectory(self._system, obs, ctrls)
        else:
            if idx < -len(self) or idx >= len(self):
                raise IndexError("Time index out of range.")
            return TimeStep(self._obs[idx,:], self._ctrls[idx,:])

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            if isinstance(idx[0], int):
                if idx[0] < -len(self) or idx[0] >= len(self):
                    raise IndexError("Time index out of range.")
            if idx[1] in self._system.observations:
                obs_idx = self._system.observations.index(idx[1])
                self._obs[idx[0], obs_idx] = val
            elif idx[1] in self._system.controls:
                ctrl_idx = self._system.controls.index(idx[1])
                self._ctrls[idx[0], ctrl_idx] = val
            else:
                raise IndexError("Unknown label")
        elif isinstance(idx, int):
            raise IndexError("Cannot assign to time steps.")
        else:
            raise IndexError("Unknown index type")

    def __len__(self):
        return len(self._obs)

    def __str__(self):
        return "Trajectory, length={}, system={}".format(len(self._obs),self._system)

    @property
    def system(self) -> System:
        """
        Get trajectory System object.
        """
        return self._system

    @property
    def obs(self) -> np.ndarray:
        """
        Get trajectory observations as a numpy array of
        shape (size, self.system.obs_dim)
        """
        return self._obs

    @obs.setter
    def obs(self, obs : np.ndarray):
        if obs.shape != self._obs.shape:
            raise ValueError("obs is wrong shape")
        self._obs = obs[:]

    @property
    def ctrls(self) -> np.ndarray:
        """
        Get trajectory controls as a numpy array of
        shape (size, self.system.ctrl_dim)
        """
        return self._ctrls

    @ctrls.setter
    def ctrls(self, ctrls : np.ndarray):
        if ctrls.shape != self._ctrls.shape:
            raise ValueError("ctrls is wrong shape")
        self._ctrls = ctrls[:]

    @property
    def times(self) -> np.ndarray:
        """Returns the array of times on which the trajectory samples are
        defined, starting at 0.
        """
        return np.array(range(len(self)))*self.system.dt

    def clone(self):
        return Trajectory(self.system, np.copy(self.obs), np.copy(self.ctrls))

    def extend(self, obs, ctrls) -> 'Trajectory':
        """
        Create a new trajectory which extends an existing trajectory
        by one or more timestep.

        Note: it is expensive O(N^2) to repeatedly extend a trajectory.

        Parameters
        ----------
        obs : numpy array of shape (N, system.obs_dim)
            New observations

        ctrls : numpy array of shape (N, system.ctrl_dim)
            New controls

        Returns
        ----------
        traj : Trajectory
            The extended trajectory
        """
        newobs = np.concatenate([self.obs, obs])
        newctrls = np.concatenate([self.ctrls, ctrls])
        return Trajectory(self.system, newobs, newctrls)
    
    def project(self,obs_dims,ctrl_dims=None):
        """Projeccts this trajectory onto a subset of dimensions."""
        if ctrl_dims is None:
            ctrl_dims = range(self.system.ctrl_dim)
        obs_dims = [d if isinstance(d,int) else self.system.observations.index(d) for i,d in enumerate(obs_dims)]
        ctrls_dims = [d if isinstance(d,int) else self.system.controls.index(d) for i,d in enumerate(ctrl_dims)]
        newsys = System([self.system.observations[i] for i in obs_dims],[self.system.controls[i] for i in ctrl_dims],self.system.dt)
        return Trajectory(newsys,self._obs[:,obs_dims],self._ctrls[:,ctrl_dims])

    def differences(self,velocities=True,diff_ctrls=False):
        """Returns the finite differences of observations along this trajectory.

        If velocities = True, the differences will be divided by system.dt

        if diff_ctrls = True, the differences of controls will also be computed
        """
        dobs = self._obs[:,1:]-self._obs[:,:-1]
        if diff_ctrls:
            dctrls = self._ctrls[:,1:]-self._ctrls[:,:-1]
        else:
            dctrls = self._ctrls[:,:-1]
        if velocities and not self.system.discrete_time:
            dobs *= 1.0/self.system.dt
            if diff_ctrls:
                dctrls *= 1.0/self.system.dt
        return Trajectory(self.system,dobs,dctrls)

    def __add__(self,traj : 'Trajectory') -> 'Trajectory':
        if len(self) != len(traj):
            raise ValueError("Can only perform arithmetic on trajectories of the same length")
        return Trajectory(self.system,self._obs + traj._obs,self._ctrls + traj._ctrls)

    def __sub__(self,traj : 'Trajectory') -> 'Trajectory':
        if len(self) != len(traj):
            raise ValueError("Can only perform arithmetic on trajectories of the same length")
        return Trajectory(self.system,self._obs - traj._obs,self._ctrls - traj._ctrls)
    
    def __mul__(self,scale : float) -> 'Trajectory':
        if not isinstance(scale,(float,int)):
            raise TypeError("Invalid scale, must be numeric")
        return Trajectory(self.system,self._obs*scale,self._ctrls*scale)

    def __div__(self,scale : float) -> 'Trajectory':
        if not isinstance(scale,(float,int)):
            raise TypeError("Invalid scale, must be numeric")
        scale = 1.0/scale
        return Trajectory(self.system,self._obs*scale,self._ctrls*scale)

    def __rmul__(self,scale : float) -> 'Trajectory':
        return self.__mul__(scale)
class DynamicTrajectory(Trajectory):
    """
    A trajectory that can be more easily extended, where obs and ctrls are
    lists.

    Call freeze() to return a standard frozen Trajectory.
    """
    def __init__(self, system : System, obs : list = None, ctrls : list = None):
        """
        Parameters
        ----------
        system : System
            The corresponding robot system

        obs : list of list or np.ndarray
            Observations at all timesteps

        ctrls : list of list or np.ndarray
            Controls at all timesteps.
        """
        self._system = system
        if obs is None:
            obs = []
        if ctrls is None:
            ctrls = []

        # Check inputs
        if isinstance(obs,np.ndarray):
            obs = obs.tolist()
        if isinstance(ctrls,np.ndarray):
            ctrls = ctrls.tolist()
        if len(obs) != len(ctrls):
            raise ValueError("obs and ctrls do not have same number of steps, obs has {}, ctrls has {}".format((len(obs),len(ctrls))))
        for i in range(len(obs)):
            if len(obs[i]) != system.obs_dim:
                raise ValueError("obs[{}] has wrong length {}, should be {}".format(i,len(obs[i]),system.obs_dim))
            if len(ctrls[i]) != system.ctrl_dim:
                raise ValueError("ctrls[{}] has wrong length {}, should be {}".format(i,len(ctrls[i]),system.ctrl_dim))

        self._obs = obs
        self._ctrls = ctrls

    def clone(self):
        return Trajectory(self.system, copy.deepcopy(self.obs), copy.deepcopy(self.ctrls))

    def append(self, obs, ctrl):
        """Extends the current trajectory, in place."""
        if len(obs) != self.system.obs_dim:
            raise ValueError("obs has wrong length {}, should be {}".format(len(obs,self.system.obs_dim)))
        if len(ctrl) != self.system.ctrl_dim:
            raise ValueError("ctrl has wrong length {}, should be {}".format(len(ctrl,self.system.ctrl_dim)))
        self._obs.append(obs[:])
        self._ctrls.append(ctrl[:])

    def extend(self, obs, ctrls) -> 'DynamicTrajectory':
        """
        Create a new trajectory which extends an existing trajectory
        by one or more timestep.

        Parameters
        ----------
        obs : list of lists or numpy arrays
            New observations

        ctrls : list of lists or numpy arrays
            New controls

        Returns
        ----------
        traj : DynamicTrajectory
            The extended trajectory
        """
        newobs = self.obs + obs
        newctrls = self.ctrls + ctrls
        return DynamicTrajectory(self.system, newobs, newctrls)

    def freeze(self) -> Trajectory:
        return Trajectory(self.system, np.array(self._obs), np.array(self._ctrls))