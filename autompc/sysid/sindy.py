import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace
from sklearn.linear_model import Lasso
#import pysindy as ps

from ..model import Model
from ..hyper import ChoiceHyperparam, MultiChoiceHyperparam, FloatRangeHyperparam

class SINDy(Model):
    def __init__(self, system):
        super().__init__(system)
        self.basis_functions = MultiChoiceHyperparam(["poly3", "trig"])


    def train(self, trajs):
        self.model = ps.SINDy(feature_library=ps.FourierLibrary(n_frequencies=1), discrete_time=True)
        X = [traj.obs for traj in trajs]
        U = [traj.ctrls for traj in trajs]
        self.model.fit(X, u=U, multiple_trajectories=True)

    def pred(self, traj, latent=None):
        # Compute transformed state x
        u = traj[-1].ctrl
        x = self._transform_state(traj[-1].obs)

        xnew = self.A @ x + self.B @ u

        # Transform to original state space xpred
        xpred = xnew[:self.system.obs_dim]

        return xpred, None

    def pred_diff(self, traj, us, latent=None):
        # Compute transformed state x
        u = traj[-1].ctrl

        xnew = self.A @ x + self.B @ u

        # Transform to original state space xpred
        # Compute grad

        return xnew, None, grad

    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
