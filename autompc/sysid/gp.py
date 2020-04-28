# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..model import Model, Hyper

class GP(Model):
    def __init__(self):
        # Initialize hyperparameters and parameters to default values
        pass

    def _get_predict_array(self, xs, us):
        #TODO
        pass

    def _get_training_arrays(self, trajs):
        #TODO
        pass

    def trains(self, trajs):
        X, y = self._get_training_arrays(trajs)

        if self.kernel == "rbf":
            kernel = RBF()
        elif self.kernel == "white":
            kernel = WhiteKernel()
        else:
            raise ValueError("Kernel {} not supported".format(self.kernel))

        self.gp = GaussianProcessRegressor(kernel=kernel,
                alpha=0.0).fit(X, y)

    def __call__(self, xs, us, latent=None, ret_grad=False):
        X = self._get_predict_array(xs)
        xnew = self.gp.predict(X)
        
        if ret_grad:
            #TODO compute xgrad, ugrad
            return xnew, (xgrad, ugrad)
        else:
            return xnew

    def get_hyper_options(self):
        return {"kernel" : (Hyper.choice, set(["rbf", "kenel"]))}

    def get_hypers(self):
        return {"kernel" : self.kernel} 

    def set_hypers(self, hypers):
        self.kernel = hypers["kernel"]

    def get_parameters(self):
        return {"gp_params" : self.gp.get_params(True)}

    def set_parameters(self, params):
        if self.kernel == "rbf":
            kernel = RBF()
        elif self.kernel == "white":
            kernel = WhiteKernel()
        else:
            raise ValueError("Kernel {} not supported".format(self.kernel))

        self.gp = GaussianProcessRegressor(kernel=kernel,
                alpha=0.0)
        self.gp.set_params(**params["gp_params"])


