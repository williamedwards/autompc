# Created by William Edwards (wre2@illinois.edu)

import numpy as np
from collections import namedtuple

def zeros(system, size):
    obs = np.zeros((size, system.obs_dim))
    ctrls = np.zeros((size, system.ctrl_dim))
    return Trajectory(system, size, obs, ctrls)

def empty(system, size):
    obs = np.empty((size, system.obs_dim))
    ctrls = np.empty((size, system.ctrl_dim))
    return Trajectory(system, size, obs, ctrls)

def extend(traj, obs, ctrls):
    newobs = np.concatenate([traj.obs, obs])
    newctrls = np.concatenate([traj.ctrls, ctrls])
    newtraj = Trajectory(traj.system, newobs.shape[0],
            newobs, newctrls)
    return newtraj

class Trajectory:
    def __init__(self, system, size, obs, ctrls):
        self._system = system
        self._size = size

        # Check inputs
        if obs.shape != (size, system.obs_dim):
            raise ValueError("obs is wrong shape")
        if ctrls.shape != (size, system.ctrl_dim):
            raise ValueError("ctrls is wrong shape")

        self._obs = obs
        self._ctrls = ctrls

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
            return namedtuple("TimeStep", "obs ctrl")(self._obs[idx,:], 
                    self._ctrls[idx,:])

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

    @property
    def system(self):
        return self._system

    @property
    def size(self):
        return self._size

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, obs):
        if obs.shape != (self._size, self._system.obs_dim):
            raise ValueError("obs is wrong shape")
        self._obs = obs[:]

    @property
    def ctrls(self):
        return self._ctrls

    @ctrls.setter
    def ctrls(self, ctrls):
        if ctrls.shape != (self._size, self._system.ctrl_dim):
            raise ValueError("ctrls is wrong shape")
        self._ctrls = ctrls[:]
