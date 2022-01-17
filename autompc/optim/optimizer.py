# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from pdb import set_trace

class Optimizer(ABC):
    def __init__(self, system, ocp):
        self.system = system

    def get_config_space(self):
        return get_default_config_space()

    @abstractmethod
    def get_default_config_space(self):
        raise NotImplementedError

    @abstractmethod
    def run(self, state):
        """
        Run the controller for a given time step

        Parameters
        ----------
            state : numpy array of size self.state_dim
                Current controller state
        Returns
        -------
            ctrl : numpy array of size self.system.ctrl_dim
                Next control input
        """
        raise NotImplementedError

    def reset(self):
        """
        Re-initialize the optimizer. For optimizers which
        cache previous results to warm-start optimization, this
        will clear the cache.
        """
        pass

    def set_model(self, model):
        self.model = model
    
    def set_ocp(self, ocp):
        self.ocp = ocp

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state):
        raise NotImplementedError