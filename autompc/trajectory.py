# Created by William Edwards (wre2@illinois.edu)

import numpy as np
from collections import namedtuple
from .system import System



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
    def __init__(self, system : System, size : int, obs : np.ndarray, ctrls : np.ndarray):
        """
        Parameters
        ----------
        system : System
            The corresponding robot system

        size : int
            Number of time steps in the trajectrory

        obs : numpy array of shape (size, system.obs_dim)
            Observations at all timesteps

        ctrls : numpy array of shape (size, system.ctrl_dim)
            Controls at all timesteps.
        """
        self._system = system
        self._size = size

        # Check inputs
        if obs.shape != (size, system.obs_dim):
            raise ValueError("obs is wrong shape, should be {}, got {}".format((size,system.obs_dim),obs.shape))
        if ctrls.shape != (size, system.ctrl_dim):
            raise ValueError("ctrls is wrong shape, should be {}, got {}".format((size,system.ctrl_dim),ctrls.shape))

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
        return Trajectory(system, size, obs, ctrls)

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
        return Trajectory(system, size, obs, ctrls)


    def __eq__(self, other):
        return (self._system == other.system
                and self._size == other._size
                and np.array_equal(self._obs, other._obs)
                and np.array_equal(self._ctrls, other._ctrls))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if (not isinstance(idx[0], slice) and (idx[0] < -self.size 
                    or idx[0] >= self.size)):
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
            #if idx.start < -self.size or idx.stop >= self.size:
            #    raise IndexError("Time index out of range.")
            obs = self._obs[idx, :]
            ctrls = self._ctrls[idx, :]
            return Trajectory(self._system, obs.shape[0], obs, ctrls)
        else:
            if idx < -self.size or idx >= self.size:
                raise IndexError("Time index out of range.")
            return TimeStep(self._obs[idx,:], self._ctrls[idx,:])

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            if isinstance(idx[0], int):
                if idx[0] < -self.size or idx[0] >= self.size:
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
        return self._size

    def __str__(self):
        return "Trajectory, length={}, system={}".format(self._size,self._system)

    @property
    def system(self) -> System:
        """
        Get trajectory System object.
        """
        return self._system

    @property
    def size(self) -> int:
        """
        Number of time steps in trajectory
        """
        return self._size

    @property
    def obs(self):
        """
        Get trajectory observations as a numpy array of
        shape (size, self.system.obs_dim)
        """
        return self._obs

    @obs.setter
    def obs(self, obs):
        if obs.shape != (self._size, self._system.obs_dim):
            raise ValueError("obs is wrong shape")
        self._obs = obs[:]

    @property
    def ctrls(self):
        """
        Get trajectory controls as a numpy array of
        shape (size, self.system.ctrl_dim)
        """
        return self._ctrls

    @ctrls.setter
    def ctrls(self, ctrls):
        if ctrls.shape != (self._size, self._system.ctrl_dim):
            raise ValueError("ctrls is wrong shape")
        self._ctrls = ctrls[:]

    def clone(self):
        return Trajectory(self.system, self.size, np.copy(self.obs), np.copy(self.ctrls))

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
        newtraj = Trajectory(self.system, newobs.shape[0],
                newobs, newctrls)
        return newtraj