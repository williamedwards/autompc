# Created by William Edwards (wre2@illinois.edu)

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from .model import Model, ModelFactory
#from ..hyper import IntRangeHyperparam

class ARXFactory(ModelFactory):
    R"""
    Autoregression with Exogenous Variable (ARX) learns the dynamics as
    a linear function of the last :math:`k` observations and controls.
    That is

    .. math::
        x_{t+1} = [x_t, \ldots x_{t-k+1}, u_t, \ldots, u_{t-k+1}] \theta

    The model is trained least-squared linear regression.

    Hyperparameters:

    - *history* (Type: int, Low: 1, High: 10, Default: 4): Size of history window
      for ARX model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Model = ARX
        self.name = "ARX"

    def get_configuration_space(self):
        cs = ConfigurationSpace()
        history = UniformIntegerHyperparameter(name='history', 
                lower=1, upper=10, default_value=4)
        cs.add_hyperparameter(history)
        return cs

class ARX(Model):
    def __init__(self, system, history):
        super().__init__(system)
        self.k = history

    def _get_feature_vector(self, traj, t=None):
        k = self.k
        if t is None:
            t = len(traj)

        feature_elements = [traj[t-1].obs]
        for i in range(t-2, t-k-1, -1):
            if i >= 0:
                feature_elements += [traj[i].obs, traj[i].ctrl]
            else:
                feature_elements += [traj[0].obs, traj[0].ctrl]
        feature_elements += [np.ones(1), traj[t-1].ctrl]
        return np.concatenate(feature_elements)

    def _get_all_feature_vectors(self, traj):
        k = self.k
        feature_vectors = np.zeros((len(traj), k*(self.system.obs_dim + self.system.ctrl_dim)+1))
        feature_vectors[:,:self.system.obs_dim] = traj.obs
        j = self.system.obs_dim
        for i in range(1, k, 1):
            feature_vectors[:,j:j+self.system.obs_dim] = np.concatenate([traj.obs[:1,:]]*i + [traj.obs[:-i, :]])
            j += self.system.obs_dim
            feature_vectors[:,j:j+self.system.ctrl_dim] = np.concatenate([traj.ctrls[:1,:]]*i + [traj.ctrls[:-i, :]])
            j += self.system.ctrl_dim
        feature_vectors[:,-(self.system.ctrl_dim+1)] = 1
        feature_vectors[:,-self.system.ctrl_dim:] = traj.ctrls

        return feature_vectors

    def _get_fvec_size(self):
        k = self.k
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

    def update_state(self, state, new_ctrl, new_obs):
        # Shift the targets
        newstate = self.A @ state + self.B @ new_ctrl
        newstate[:self.system.obs_dim] = new_obs

        return newstate

    def traj_to_state(self, traj):
        return self._get_feature_vector(traj)[:-self.system.ctrl_dim]

    def traj_to_states(self, traj):
        return self._get_all_feature_vectors(traj)[:, :-self.system.ctrl_dim]

    def state_to_obs(self, state):
        return state[0:self.system.obs_dim]

    def train(self, trajs, silent=False):
        matrix, targets = self._get_training_matrix_and_targets(trajs)

        coeffs = np.zeros((self.system.obs_dim, self._get_fvec_size()))
        for i in range(targets.shape[1]):
            res, _, _,  _ = la.lstsq(matrix, targets[:,i], rcond=None)
            coeffs[i,:] = res

        # First we construct the system matrices
        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, self.system.ctrl_dim))

        # Constant term
        A[-1,-1] = 1.0

        # Shift history
        n = self.system.obs_dim
        l = self.system.ctrl_dim
        m = self.system.obs_dim + self.system.ctrl_dim
        k = self.k

        if k > 1:
            A[n : 2*n, 0 : n] = np.eye(n)
        for i in range(k-2):
            A[(i+1)*m+n : (i+2)*m+n, i*m+n : (i+1)*m+n] = np.eye(m)

        # Predict new observation
        A[0 : n, :] = coeffs[:, :-l]

        # Add new control
        B[0 : n, :] = coeffs[:, -l:]
        B[2*n : 2*n + l, :] = np.eye(l)

        self.A, self.B = A, B


    def pred(self, state, ctrl):
        statenew = self.A @ state + self.B @ ctrl

        return statenew

    def pred_batch(self, states, ctrls):
        statesnew = self.A @ states.T + self.B @ ctrls.T

        return statesnew.T

    def pred_diff(self, state, ctrl):
        statenew = self.A @ state + self.B @ ctrl

        return statenew, self.A, self.B

    def to_linear(self):
        return self.A, self.B

    @property
    def state_dim(self):
        return self._get_fvec_size() - self.system.ctrl_dim


    def get_parameters(self):
        return {"coeffs" : np.copy(self.coeffs)}

    def set_parameters(self, params):
        self.coeffs = np.copy(params["coeffs"])


