# Created by William Edwards (wre2@illinois.edu)

from abc import ABC, abstractmethod
from pdb import set_trace

from .hyper import Hyperparam

class Controller(ABC):
    def __init__(self, system):
        self.system = system

    @abstractmethod
    def run(self, traj, latent=None):
        """
        Parameters
        ----------
            traj : Trajectory
                State and control history up to present time.
            latent : Arbitrary python object
                Latent model data. Can be arbitrary
                python object. None should be passed for first time
                step. Used only to avoid redundant computation.
        Returns
        -------
            u : Next control input
            latent : Arbitrary python object
                Data used for latent computation. May be none.
        """
        raise NotImplementedError

    def run_diff(self, traj, latent=None):
        """
        Parameters
        ----------
            traj : Trajectory
                State and control history up to present time.
            latent : Arbitrary python object
                Latent model data. Can be arbitrary
                python object. None should be passed for first time
                step. Used only to avoid redundant computation.
        Returns
        -------
            u : Next control input
            latent : Arbitrary python object
                Data used for latent computation. May be none.
            grad :
                Gradient of control wrt state and control history.
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



    @property
    def is_diff(self):
        """
        Returns true for differentiable models.
        """
        return not self.run_diff.__func__ is Controller.run_diff
