# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
import copy
from pdb import set_trace

class Optimizer(ABC):
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
    def step(self, state):
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

    @abstractmethod
    def is_compatible(self, model, ocp):
        """
        Check if an optimizer is compatible with a given model
        or ocp/ocp factory.

        Parameters
        ----------
        model : Model
            Model to check compatibility.
        
        ocp : OCP
            OCP to check compatibility
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """
        Alias of step() included for backwards compatibility.
        """
        return self.step(*args, **kwargs)

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

    def clone(self):
        return copy.deepcopy(self)