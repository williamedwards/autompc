# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from pdb import set_trace

from .hyper import Hyperparam

class Controller(ABC):
    def __init__(self, system, task, model):
        self.system = system
        self.model = model
        self.task = task

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
               Corresponding controller state
        """
        raise NotImplementedError

    #@abstractmethod
    #def update_state(self, state, new_ctrl, new_obs):
    #    """
    #    Parameters
    #    ----------
    #        state : numpy array of size self.state_dim
    #            Current controller state
    #        new_ctrl : numpy array of size self.system.ctrl_dim
    #            New control input
    #        new_obs : numpy array of size self.system.obs_dim
    #            New observation
    #    Returns
    #    -------
    #        state : numpy array of size self.state_dim
    #            Model state after observation and control
    #    """
    #    raise NotImplementedError


    @abstractmethod
    def run(self, state, new_obs):
        """
        Parameters
        ----------
            state : numpy array of size self.state_dim
                Current controller state
            new_obs : numpy array of size self.system.obs_dim
                Current observation state.
        Returns
        -------
            ctrl : Next control input
            newstate : numpy array of size self.state_dim
        """
        raise NotImplementedError
    

    @staticmethod
    @abstractmethod
    def get_configuration_space(system, task, model):
        """
        Returns the controller configuration space for the given
        system, model, and task.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_compatible(system, task, model):
        """
        Returns true if the controller is compatible with
        the given system, model, and task. Returns false
        otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def state_dim(self):
        """
        Returns the size of the model state.
        """
        raise NotImplementedError
