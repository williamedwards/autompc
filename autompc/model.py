# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from pdb import set_trace

from .hyper import Hyperparam

class Model(ABC):
    def __init__(self, system):
        self.system = system

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
        Parameters
        ----------
            state : Numpy array of size self.state_dim
                Model state
            ctrl : Numpy array of size self.system.ctrl_dim
                Control to be applied
        Returns
        -------
            state : Numpy array of size self.state_dim
                New predicted model state
        """
        raise NotImplementedError

    @abstractmethod
    def pred_diff(self, state, ctrl):
        """
        Parameters
        ----------
            state : Numpy array of size self.state_dim
                Model state
            ctrl : Numpy array of size self.system.ctrl_dim
                Control to be applied
        Returns
        -------
            state : Numpy array of size self.state_dim
                New predicted model state
            state_jac : Numpy  array of shape (self.state_dim, 
                self.state_dim).
                Gradient of predicted model state wrt to state
            ctrl_jac : Numpy  array of shape (self.state_dim, 
                self.ctrl_dim).
                Gradient of predicted model state wrt to ctrl
        """
        raise NotImplementedError

    def to_linear(self):
        """
        Returns: (A, B, state_func, cost_func)
            A, B -- Linear system matrices as Numpy arrays.
        Only implemented for linear models.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_configuration_space(system):
        """
        Returns the model configuration space.
        """
        raise NotImplementedError

    def train(self, trajs):
        """
        Parameters
        ----------
            trajs : List of pairs (xs, us)
                Training set of trajectories
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
    def is_trainable(self):
        """
        Returns true for trainable models.
        """
        return not (self.train.__func__ is Model.train
                or self.get_parameters.__func__ is Model.get_parameters
                or self.set_parameters.__func__ is Model.set_parameters)
