# Created by William Edwards (wre2@illinois.edu)

import numpy as np

from abc import ABC, abstractmethod
import copy

from ..trajectory import zeros

class Model(ABC):
    def __init__(self, system, name):
        self.system = system
        self.name = name
        self.set_config(self.get_default_config())

    def get_config_space(self):
        return self.get_default_config_space()

    @abstractmethod
    def get_default_config_space(self):
        raise NotImplementedError

    def get_default_config(self):
        return self.get_config_space().get_default_configuration()

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @abstractmethod
    def set_config(self, config):
        raise NotImplementedError

    def clone(self):
        return copy.deepcopy(self)

    @abstractmethod
    def traj_to_state(self, traj):
        """
        Parameters
        ----------
            traj : Trajectory
                State and control history up to present time
        Returns
        -------
            state : numpy array of size self.state_dim
               Corresponding model state
        """
        raise NotImplementedError

    def init_state(self, obs):
        """
        Parameters
        ----------
        """
        traj = zeros(self.system, 1)
        traj[0].obs[:] = obs
        return self.traj_to_state(traj)

    @abstractmethod
    def update_state(self, state, new_ctrl, new_obs):
        """
        Parameters
        ----------
            state : numpy array of size self.state_dim
                Current model state
            new_ctrl : numpy array of size self.system.ctrl_dim
                New control input
            new_obs : numpy array of size self.system.obs_dim
                New observation
        Returns
        -------
            state : numpy array of size self.state_dim
                Model state after observation and control
        """
        raise NotImplementedError

    @abstractmethod
    def pred(self, state, ctrl):
        """
        Run model prediction.

        Parameters
        ----------
            state : Numpy array of size self.state_dim
                Model state at time t
            ctrl : Numpy array of size self.system.ctrl_dim
                Control applied at time t
        Returns
        -------
            state : Numpy array of size self.state_dim
                Predicted model state at time t+1
        """
        raise NotImplementedError

    def pred_batch(self, states, ctrls):
        """
        Run batch model predictions.  Depending on the model, this can
        be much faster than repeatedly calling pred.

        Parameters
        ----------
            state : Numpy array of size (N, self.state_dim)
                N model input states
            ctrl : Numpy array of size (N, self.system.ctrl_dim)
                N controls
        Returns
        -------
            state : Numpy array of size (N, self.state_dim)
                N predicted states
        """
        n = self.state_dim
        m = states.shape[0]
        out = np.empty((m, n))
        for i in range(m):
            out[i,:] = self.pred(states[i,:], ctrls[i,:])
        return out

    def pred_diff(self, state, ctrl):
        """
        Run model prediction and compute gradients.

        Parameters
        ----------
            state : Numpy array of size self.state_dim
                Model state at time t
            ctrl : Numpy array of size self.system.ctrl_dim
                Control at time t
        Returns
        -------
            state : Numpy array of size self.state_dim
                Predicted model state at time t+1
            state_jac : Numpy  array of shape (self.state_dim, 
                        self.state_dim)
                Gradient of predicted model state wrt to state
            ctrl_jac : Numpy  array of shape (self.state_dim, 
                       self.ctrl_dim)
                Gradient of predicted model state wrt to ctrl
        """
        raise NotImplementedError

    def pred_diff_batch(self, states, ctrls):
        """
        Run model prediction and compute gradients in batch.

        Parameters
        ----------
            state : Numpy array of shape (N, self.state_dim)
                N input model states
            ctrl : Numpy array of size (N, self.system.ctrl_dim)
                N input controls
        Returns
        -------
            state : Numpy array of size (N, self.state_dim)
                N predicted model states
            state_jac : Numpy  array of shape (N, self.state_dim, 
                        self.state_dim)
                Gradient of predicted model states wrt to state
            ctrl_jac : Numpy  array of shape (N, self.state_dim, 
                       self.ctrl_dim)
                Gradient of predicted model states wrt to ctrl
        """
        n = self.state_dim
        m = states.shape[0]
        out = np.empty((m, n))
        state_jacs = np.empty((m, n, n))
        ctrl_jacs = np.empty((m, n, self.system.ctrl_dim))
        for i in range(m):
            out[i,:], state_jacs[i,:,:], ctrl_jacs[i,:,:] = \
                self.pred_diff(states[i,:], ctrls[i,:])
        return out, state_jacs, ctrl_jacs


    def to_linear(self):
        """
        Returns: (A, B, state_func, cost_func)
            A, B -- Linear system matrices as Numpy arrays.
        Only implemented for linear models.
        """
        raise NotImplementedError

    def train(self, trajs, silent=False):
        """
        Parameters
        ----------
            trajs : List of pairs (xs, us)
                Training set of trajectories
            silent : bool
                Silence progress bar output
        Only implemented for trainable models.
        """
        raise NotImplementedError

    def get_parameters(self):
        """
        Returns a dict containing trained model parameters.

        Only implemented for trainable models.
        """
        raise NotImplementedError

    def set_parameters(self, params):
        """
        Sets trainable model parameters from dict.

        Only implemented for trainable parameters.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def state_dim(self):
        """
        Returns the size of the model state
        """
        raise NotImplementedError


    @property
    def is_linear(self):
        """
        Returns true for linear models
        """
        return not self.to_linear.__func__ is Model.to_linear

    @property
    def is_diff(self):
        """
        Returns true for differentiable models.
        """
        return not self.pred_diff.__func__ is Model.pred_diff

    @property
    def trainable(self):
        """
        Returns true for trainable models.
        """
        return not self.train.__func__ is Model.train
