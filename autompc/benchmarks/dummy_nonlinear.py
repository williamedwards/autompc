import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace
from sklearn.linear_model import  Lasso

from .model import Model

# Simulates 2-state system
# x1[k+1] = x1[k] + x2[k]**3
# x2[k+1] = x2[k] + u

class DummyNonlinear(Model):
    def __init__(self, system):
        super().__init__(system)

    def state_dim(self):
        return 2

    def train(self, trajs):
        pass

    def traj_to_state(self, traj):
        state = np.zeros((2,))
        state[:] = traj[-1].obs[:]
        return state[:]

    def update_state(state, new_obs, new_ctrl):
        return state[:]

    def pred(self, state, ctrl):
        u = ctrl[0]
        x1, x2 = state[0], state[1]
        xpred = np.array([x1 + x2**3, x2 + u])

        return xpred

    def pred_diff(self, state, ctrl):
        u = ctrl[0]
        x1, x2 = state[0], state[1]
        xpred = np.array([x1 + x2**3, x2 + u])
        grad1 = np.array([[1.0, 3 * x2 ** 2], [0., 1.]])
        grad2 = np.array([[0.], [1.]])
        return xpred, grad1, grad2

    @staticmethod
    def get_configuration_space(system):
        """
        Returns the model configuration space.
        """
        return None
