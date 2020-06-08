import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace
from sklearn.linear_model import  Lasso

from ..model import Model
from ..hyper import ChoiceHyperparam, MultiChoiceHyperparam, FloatRangeHyperparam
from ..gradient import gradzeros

# Simulates 2-state system
# x1[k+1] = x1[k] + x2[k]**3
# x2[k+1] = x2[k] + u

class DummyNonlinear(Model):
    def __init__(self, system):
        super().__init__(system)

    def train(self, trajs):
        pass

    def pred(self, traj):
        u = traj[-1].ctrl[0]
        x1, x2 = traj[-1].obs
        xpred = [x1 + x2**3, x2 + u]

        return xpred, None

    def pred_diff(self, traj, latent=None):
        u = traj[-1].ctrl[0]
        x1, x2 = traj[-1].obs
        xpred = [x1 + x2**3, x2 + u]

        grad = gradzeros(self.system, traj.size, 2)
        grad[-1, "x1", 0] = 1.0
        grad[-1, "x2", 0] = 3 * x2**2
        grad[-1, "x2", 1] = 1.0
        grad[-1, "u", 1] = 1.0

        return xpred, None, grad

    def to_linear(self):
        # Compute state transform state_func
        # Compute cost transformer cost_func
        def state_func(traj):
            return self._transform_state(traj[-1].obs)
        def cost_func(Q, R, F=None):
            n = self.system.obs_dim
            Qt = np.zeros((self._state_size(), self._state_size()))
            Qt[:n, :n] = Q
            if F is None:
                return Qt, R
            else:
                Ft = np.zeros_like(Qt)
                Ft[:n, :n] = F
                return Qt, R, Ft
        return np.copy(self.A), np.copy(self.B), state_func, cost_func

    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
