# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from pdb import set_trace

class ControllerFactory(ABC):
    """
    The ControllerFactroy creates a controller based
    on a hyperparameter configuration.
    """
    def __init__(self, system, **kwargs):
        self.system = system
        self.kwargs = kwargs

    def __call__(self, cfg, task, model):
        """
        Returns initialized controller.

        Parameters
        ----------
        cfg : Configuration
            Hyperparameter configuration for the controller

        task : Task
            Task which controller will solve

        model : Model
            System ID model to use for optimization
        """
        controller_kwargs = cfg.get_dictionary()
        controller_kwargs.update(self.kwargs)
        controller = self.Controller(self.system, task, model, **controller_kwargs)
        return controller

    def get_configuration_space(self):
        """
        Returns the controller ConfigurationSpace.
        """
        raise NotImplementedError

class Controller(ABC):
    def __init__(self, system, task, model):
        """
        Initialize the controller.

        Parameters
        ----------
        system : System
            Robot system to control

        task : Task
            Task which controller will solve

        model : Model
            System ID model to use for optimization
        """
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
               Corresponding controller state.  This is frequently,
               but not always, equal to the underlying systme ID
               model state.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, state, new_obs):
        """
        Run the controller for a given time step

        Parameters
        ----------
            state : numpy array of size self.state_dim
                Current controller state
            new_obs : numpy array of size self.system.obs_dim
                Current observation state.
        Returns
        -------
            ctrl : numpy array of size self.system.ctrl_dim
                Next control input
            newstate : numpy array of size self.state_dim
                New controller state
        """
        raise NotImplementedError

    def reset(self):
        """
        Re-initialize the controller. For controllers which
        cache previous results to warm-start optimization, this
        will clear the cache.
        """
        pass
    
    # @staticmethod
    # @abstractmethod
    # def is_compatible(system, task, model):
    #     """
    #     Returns true if the controller is compatible with
    #     the given system, model, and task. Returns false
    #     otherwise.
    #     """
    #     raise NotImplementedError

    @property
    @abstractmethod
    def state_dim(self):
        """
        Returns the size of the controller state.
        """
        raise NotImplementedError
