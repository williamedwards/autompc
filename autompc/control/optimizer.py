# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from pdb import set_trace

from ..utils import apply_partial

class OptimizerFactory(ABC):
    """
    Factory to create MPC optimizers.
    """
    def __init__(self, system, **kwargs):
        """
        Parameters
        ----------
            system : System
                System to be controlled
            **kwargs
                Extra keyword arguments to be passed to
                optimizer initialization.
        """
        self.system = system
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return apply_partial(self, self.Optimizer, 3, args, kwargs)

    def create(self, cfg, model, ocp):
        """
        Returns the initialized optimizer.

        Parameters
        ----------
            cfg : Configuration
                Configuration of optimizer hyperparamters
            model : Model
                System ID model used for planning
            ocp : ControlProblem
                Optimal control problem to solve
        """
        optimizer_args = cfg.get_dictinary()
        optimizer_args.update(self.kwargs)
        optimizer = self.Optimizer(self.system, model, ocp,
            **optimizer_args)

        return optimizer

class Optimizer:
    def __init__(system, model, ocp):
        """
        Initialize the controller.

        Parameters
        ----------
        system : System
            Robot system to contorl

        ocp : OCP
            Optimal control problem to be solved

        model : Model
            System ID model to use for optimization
        """
        self.system = system
        self.model = model
        self.task = task

    @abstractmethod
    def run(self, model_state):
        """
        Run the optimizer to produce a control.

        Parameters
        ----------
            state : numpy array of size self.model.state_dim
                Current model state
        Returns
        -------
            ctrl : numpy array of size self.system.ctrl_dim
                Next control input
            newstate : numpy array of size self.state_dim
                New controller state
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """
        Returns the controller state as some pickle-able object.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state):
        """
        Sets the controller state.
        """
        raise NotImplementedError