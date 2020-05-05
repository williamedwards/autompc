# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from ..model import Model, Hyper

class ARX(Model):
    def __init__(self):
        # Initialize hyperparameters and parameters to default values
        pass

    def _get_feature_vector(self, xs, us, t):
        # TODO
        pass

    def _get_training_matrix_and_target(self, trajs):
        # TODO
        pass

    def train(self, trajs):
        matrix, targets = self._get_training_matrix_and_targets(self, trajs)

        self.coeffs = []
        for target in targets:
            coeffs, _, _,  _ = la.lstsq(matrix, target)
            self.coeffs.append(coeffs)

    def pred(self, xs, us, latent=None):
        fvec = self._get_feature_vector(xs, us, len(xs)-1)
        xnew = np.zeros(xs.shape[1])

        for i in range(xs.shape[1]):
            xnew[i] = self.coeffs[i] @ fvec

        return xnew, None

    def pred_diff(self, xs, us, latent=None):
        fvec = self._get_feature_vector(xs, us, len(xs)-1)
        xnew = np.zeros(xs.shape[1])

        for i in range(xs.shape[1]):
            xnew[i] = self.coeffs[i] @ fvec

        xgrad = np.zeros(xs.shape)
        ugrad = np.zeros(us.shape)
        for i in range(self.k):
            xgrad[-i] = fvec[off1:off2]
            ugrad[-i] = fvec[off3:off4]
        return xnew, None, (xgrad, ugrad)

    def to_linear(self):
        pass


    def get_hyper_options(self):
        return {"k" : (Hyper.int_range, (1, float("inf")))}

    def get_hypers(self):
        return {"k" : self.k} 

    def set_hypers(self, hypers):
        self.k = hypers["k"]

    def get_parameters(self):
        return {"coeffs" : np.copy(self.coeffs)}

    def set_parameters(self, params):
        self.coeffs = np.copy(params["coeffs"])


