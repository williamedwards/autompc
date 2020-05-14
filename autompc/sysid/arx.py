# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from pdb import set_trace

from ..model import Model
from ..hyper import IntRangeHyperparam

class ARX(Model):
    def __init__(self, system):
        super().__init__(system)
        self.k = IntRangeHyperparam((1, 10))
        #TODO add regularizaition

    def _get_feature_vector(self, traj, t=None):
        k = self.k.value
        if t is None:
            t = len(traj)

        feature_elements = [np.ones(1)]
        for i in range(t-k, t):
            if i >= 0:
                feature_elements += [traj[i].obs, traj[i].ctrl]
            else:
                feature_elements += [traj[0].obs, traj[0].ctrl]
        return np.concatenate(feature_elements)

    def _get_fvec_size(self):
        k = self.k.value
        return 1 + k*self.system.obs_dim + k*self.system.ctrl_dim
        

    def _get_training_matrix_and_targets(self, trajs):
        nsamples = sum([len(traj) for traj in trajs])
        matrix = np.zeros((nsamples, self._get_fvec_size()))
        targets = np.zeros((nsamples, self.system.obs_dim))

        i = 0
        for traj in trajs:
            for t in range(1, len(traj)):
                matrix[i, :] = self._get_feature_vector(traj, t)
                targets[i, :] = traj[t].obs
                i += 1

        return matrix, targets

    def train(self, trajs):
        matrix, targets = self._get_training_matrix_and_targets(trajs)

        self.coeffs = np.zeros((self.system.obs_dim, self._get_fvec_size()))
        for i in range(targets.shape[1]):
            coeffs, _, _,  _ = la.lstsq(matrix, targets[:,i], rcond=None)
            self.coeffs[i,:] = coeffs

    def pred(self, traj, latent=None):
        fvec = self._get_feature_vector(traj)
        xnew = self.coeffs @ fvec

        return xnew, None

    def pred_diff(self, traj, latent=None):
        #TODO Implement this
        raise NotImplementedError

        fvec = self._get_feature_vector(traj)
        xnew = np.zeros(traj.system.obs_dim)

        for i in range(traj.system.obs_dim):
            xnew[i] = self.coeffs[i] @ fvec

        # Compute grad
        
        return xnew, None, grad

    def to_linear(self):
        # First we construct the system matrices
        A = np.zeros((self._get_fvec_size(), self._get_fvec_size()))
        B = np.zeros((self._get_fvec_size(), self.system.ctrl_dim))

        # Constant term
        A[0,0]

        # Shift history
        m = self.system.obs_dim + self.system.ctrl_dim
        k = self.k.value
        for i in range(k-1):
            A[1 + i*m : 1 + (i+1)*m, 1 + (i+1)*m : 1 + (i+2)*m] = np.eye(m)

        # Predict new observation
        A[1 + (k-1)*m : 1 + (k-1)*m + self.system.obs_dim, :] = self.coeffs

        # Add new control
        B[1 + k*m - self.system.ctrl_dim : 1 + k*m, :] = np.eye(self.system.ctrl_dim)

        def state_func(traj, t=None):
            return self._get_feature_vector(traj, t)

        def cost_func(Q, R):
            Qnew = np.zeros(A.shape)
            Qnew[-self.system.obs_dim:, -self.system.obs_dim:] = Q
            Rnew = R.copy()
            return Qnew, Rnew

        return A, B, state_func, cost_func



    def get_parameters(self):
        return {"coeffs" : np.copy(self.coeffs)}

    def set_parameters(self, params):
        self.coeffs = np.copy(params["coeffs"])


