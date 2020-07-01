# Created by William Edwards (wre2@illinois.edu)

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from ..model import Model
#from ..hyper import IntRangeHyperparam

class ARX(Model):
    def __init__(self, system, horizon):
        super().__init__(system)
        self.k = horizon

    @staticmethod
    def get_configuration_space(system):
        cs = ConfigurationSpace()
        horizon = UniformIntegerHyperparameter(name='horizon', 
                lower=1, upper=10, default_value=4)
        cs.add_hyperparameter(horizon)
        return cs
        

    def _get_feature_vector(self, traj, t=None):
        k = self.k.value
        if t is None:
            t = len(traj)

        feature_elements = []
        for i in range(t-1, t-k-1, -1):
            if i >= 0:
                feature_elements += [traj[i].obs, traj[i].ctrl]
            else:
                feature_elements += [traj[0].obs, traj[0].ctrl]
        feature_elements += [np.ones(1)]
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

    def update_state(self, state, new_obs, new_ctrl):
        # Shift the targets
        m = self.system.obs_dim + self.system.ctrl_dim
        k = self.k.value
        newstate = np.zeros(self._get_fvec_size())
        for i in range(k-1):
            newstate[(i+1)*m : (i+2)*m] = state[i*m : (i+1)*m]
        newstate[:self.system.obs_dim] = new_obs
        newstate[self.system.obs_dim : 
                self.system.obs_dim + self.system.ctrl_dim] = new_ctrl

        return newstate

    def traj_to_state(self, traj):
        return self._get_feature_vector(traj)

    def state_to_obs(self, state):
        return state[0:self.system.obs_dim]

    def train(self, trajs):
        matrix, targets = self._get_training_matrix_and_targets(trajs)

        coeffs = np.zeros((self.system.obs_dim, self._get_fvec_size()))
        for i in range(targets.shape[1]):
            res, _, _,  _ = la.lstsq(matrix, targets[:,i], rcond=None)
            coeffs[i,:] = res

        # First we construct the system matrices
        A = np.zeros((self._get_fvec_size(), self._get_fvec_size()))
        B = np.zeros((self._get_fvec_size(), self.system.ctrl_dim))

        # Constant term
        A[-1,-1] = 1.0

        # Shift history
        m = self.system.obs_dim + self.system.ctrl_dim
        k = self.k.value
        for i in range(k-1):
            A[(i+1)*m : (i+2)*m, i*m : (i+1)*m] = np.eye(m)

        # Predict new observation
        A[0 : self.system.obs_dim, :] = coeffs

        # Add new control
        B[self.system.obs_dim :  self.system.obs_dim + self.system.ctrl_dim, 
                :] = np.eye(self.system.ctrl_dim)

        self.A, self.B = A, B


    def pred(self, state, ctrl):
        statenew = self.A @ state + self.B @ ctrl

        return statenew

    def pred_diff(self, state, ctrl):
        statenew = self.A @ state + self.B @ ctrl

        return statenew, ctrl

    def to_linear(self):
        return self.A, self.B

    @property
    def state_dim(self):
        return self._get_fvec_size()


    def get_parameters(self):
        return {"coeffs" : np.copy(self.coeffs)}

    def set_parameters(self, params):
        self.coeffs = np.copy(params["coeffs"])


