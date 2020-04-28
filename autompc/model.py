# Created by William Edwards (wre2@illinois.edu)

from enum import Enum

class Hyper(Enum):
    """
    Enumeration for hyperparameter types.
    """
    float_range = 1
    int_range = 2
    boolean = 3
    choice = 4

class Model:
    def __call__(self, xs, us, latent=None, ret_grad=False):
        """
        Parameters
        ----------
            xs : (Numpy array)
                States up to time t.
            us : (Numpy array)
                Controls up to time t.
            latent : Arbitrary python object
                Latent model data. Can be arbitrary
                python object. None should be passed for first time
                step. Used only to avoid redundant computation.
            ret_grad : bool
                If true, function returns gradient info.
                A true value will result in a NotImplementedError if the
                model is not differentiable.
        Returns
        -------
            x : Numpy array
                state at time (t+1)
            latent : Arbitrary python object
                Data used for latent computation. May be none.
            grad : Numpy array
                Gradient of prediction wrt state and control history.
        """
        raise NotImplementedError

    def get_linear_system(self):
        """
        Returns: (A, B)
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
        raise NotImplementedError

    def get_hypers(self):
        """
        Returns a dict containing hyperaparameter values. Only implemented
        for trainable models. The keys is the hyperparameter name and the
        value is the value.
        """
        raise NotImplementedError

    def set_hypers(self, hypers):
        """
        Parameters
        ----------
            hypers : dict
                A dict containing hyperparameter names and values to
                be updated. Any hyperparameter not contained in the dict is
                left unchanged.
        Only implemented for trainable models.
        """
        raise NotImplementedError

    def train(self, trajs):
        """
        Parameters
        ----------
            trajs : List of Numpy arrays
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

    def get_parameters(self, params):
        """
        Sets trainable model parameters from dict.

        Only implemented for trainable parameters.
        """
        raise NotImplementedError
