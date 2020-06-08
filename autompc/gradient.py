# Created by William Edwards (wre2@illinois.edu)

import numpy as np
from collections import namedtuple

def gradzeros(system, size, grad_dim):
    gradobs = np.zeros((size, system.obs_dim, grad_dim))
    gradctrls = np.zeros((size, system.ctrl_dim, grad_dim))
    return Gradient(system, size, grad_dim, gradobs, gradctrls)

def gradempty(system, size, grad_dim):
    gradobs = np.empty((size, system.obs_dim, grad_dim))
    gradctrls = np.empty((size, system.ctrl_dim, grad_dim))
    return Gradient(system, size, grad_dim, gradobs, gradctrls)

#def extend(traj, obs, ctrls):
#    newobs = np.concatenate([traj.obs, obs])
#    newctrls = np.concatenate([traj.ctrls, ctrls])
#    newtraj = Trajectory(traj.system, newobs.shape[0],
#            newobs, newctrls)
#    return newtraj

class Gradient:
    def __init__(self, system, size, grad_dim, gradobs, gradctrls):
        self._system = system
        self._size = size
        self._grad_dim = grad_dim

        # Check inputs
        if gradobs.shape != (size, system.obs_dim, grad_dim):
            raise ValueError("obs is wrong shape")
        if gradctrls.shape != (size, system.ctrl_dim, grad_dim):
            raise ValueError("ctrls is wrong shape")

        self._gradobs = gradobs
        self._gradctrls = gradctrls

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if (not isinstance(idx[0], slice) and (idx[0] < -self.size 
                    or idx[0] >= self.size)):
                raise IndexError("Time index out of range.")
            if (not isinstance(idx[2], slice) and (idx[2] < -self.grad_dim 
                    or idx[2] >= self.grad_dim)):
                raise IndexError("Gradient index out of range.")
            if idx[1] in self._system.observations:
                obs_idx = self._system.observations.index(idx[1])
                return self._gradobs[idx[0], obs_idx, idx[2]]
            elif idx[1] in self._system.controls:
                ctrl_idx = self._system.controls.index(idx[1])
                return self._gradctrls[idx[0], ctrl_idx, idx[2]]
            else:
                raise IndexError("Unknown label")
        elif isinstance(idx, int):
            if idx < -self.size or idx >= self.size:
                raise IndexError("Time index out of range.")
            return namedtuple("TimeStep", "gradobs gradctrl")(self._gradobs[idx,:], 
                    self._gradctrls[idx,:])
        elif isinstance(idx, slice):
            #if idx.start < -self.size or idx.stop >= self.size:
            #    raise IndexError("Time index out of range.")
            gradobs = self._gradobs[idx, :, :]
            gradctrls = self._gradctrls[idx, :, :]
            return Trajectory(self._system, obs.shape[0], gradobs, gradctrls)
        else:
            raise IndexError("Unknown index type")

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            if idx[0] < -self.size or idx[0] >= self.size:
                raise IndexError("Time index out of range.")
            if idx[2] < -self.grad_dim or idx[2] >= self.grad_dim:
                raise IndexError("Gradient index out of range.")
            if idx[1] in self._system.observations:
                obs_idx = self._system.observations.index(idx[1])
                self._gradobs[idx[0], obs_idx, idx[2]] = val
            elif idx[1] in self._system.controls:
                ctrl_idx = self._system.controls.index(idx[1])
                self._gradctrls[idx[0], ctrl_idx, idx[2]] = val
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
    def grad_dim(self):
        return self._grad_dim

    @property
    def gradobs(self):
        return self._gradobs

    @gradobs.setter
    def gradobs(self, gradobs):
        if gradobs.shape != (self._size, self._system.obs_dim, self._grad_dim):
            raise ValueError("obs is wrong shape")
        self._gradobs = gradobs[:]

    @property
    def gradctrls(self):
        return self._gradctrls

    @gradctrls.setter
    def gradctrls(self, gradctrls):
        if gradctrls.shape != (self._size, self._system.ctrl_dim, self._grad_dim):
            raise ValueError("ctrls is wrong shape")
        self._gradctrls = gradctrls[:]
