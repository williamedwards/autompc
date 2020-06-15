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

    #@abstractmethod
    #def state_to_obs(self, obs):
    #    """
    #    Parameters
    #    ----------
    #        state : numpy array of size self.state_dim
    #    Returns
    #    -------
    #        obs : numpy array of size self.system.obs_dim
    #    """
    #    raise NotImplementedError

    @abstractmethod
    def update_state(self, state, new_obs, new_ctrl):
        """
        Parameters
        ----------
            state : numpy array of size self.state_dim
                Current model state
            new_obs : numpy array of size self.system.obs_dim
                New observation
            new_ctrl : numpy array of size self.system.ctrl_dim
                New control input
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
            grad : Numpy  array of shape (self.state_dim, self.state_dim)
                Gradient of predicted model state
        """
        raise NotImplementedError

    def to_linear(self):
        """
        Returns: (A, B, state_func, cost_func)
            A, B -- Linear system matrices as Numpy arrays.
        Only implemented for linear models.
        """
        raise NotImplementedError

    def get_hyper_options(self):
        """
        Returns a dict containing available hyperparameter options. Only
        implemented for trainable models. The key is the hyperparameter name
        and the value is a tuple of the form (hyper, option). If hyper in
        [Hyper.float_range, Hyper.int_range], option is a tuple of the form
        [lower_bound, upper_bound]. If hyper is Hyper.choice, option is a set
        of available choices.  If hyper is Hyper.boolean, option is None.
        """
        hyperopts = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, Hyperparam):
                hyperopts[k] = (v.type, v.options)
        return hyperopts

    def get_hypers(self):
        """
        Returns a dict containing hyperaparameter values. Only implemented
        for trainable models. The keys is the hyperparameter name and the
        value is the value.
        """
        hypers = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, Hyperparam):
                hypers[k] = v.value
        return hypers

    def set_hypers(self, **hypers):
        """
        Parameters
        ----------
            hypers : dict
                A dict containing hyperparameter names and values to
                be updated. Any hyperparameter not contained in the dict is
                left unchanged.
        Only implemented for trainable models.
        """
        for k, v in hypers.items():
            if k in self.__dict__ and isinstance(self.__dict__[k], Hyperparam):
                self.__dict__[k].value = v


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
