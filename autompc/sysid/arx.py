# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from ..model import Model
from ..hyper import IntRangeHyperparam

class ARX(Model):
    def __init__(self, system):
        super().__init__(system)
        self.k = IntRangeHyperparam((1, 10))

    def _get_feature_vector(self, xs, us):
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

    def pred(self, traj, latent=None):
        fvec = self._get_feature_vector(traj)
        xnew = np.zeros(traj.system.obs_dim)

        for i in range(traj.system.obs_dim):
            xnew[i] = self.coeffs[i] @ fvec

        return xnew, None

    def pred_diff(self, traj, latent=None):
        fvec = self._get_feature_vector(traj)
        xnew = np.zeros(traj.system.obs_dim)

        for i in range(traj.system.obs_dim):
            xnew[i] = self.coeffs[i] @ fvec

        # Compute grad
        
        return xnew, None, grad

    def to_linear(self):
        pass

    def get_parameters(self):
        return {"coeffs" : np.copy(self.coeffs)}

    def set_parameters(self, params):
        self.coeffs = np.copy(params["coeffs"])


