"""
This file defines a basic controller and serves as AbstractClass
"""
from .model import Model


class Controller(object):
    def __call__(self, x):
        """
        Parameters
        ----------
            x : (Numpy array)
                States at current time step.
        Returns
        -------
            u : Numpy array
                Control output at current step
        """
        raise NotImplementedError
