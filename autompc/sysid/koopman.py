import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace
from sklearn.linear_model import  Lasso

from ..model import Model
from ..hyper import ChoiceHyperparam, MultiChoiceHyperparam, FloatRangeHyperparam

class Koopman(Model):
    def __init__(self, system):
        super().__init__(system)
        self.method = ChoiceHyperparam(["lstsq", "lasso", "stableAB"])

        self.basis_functions = MultiChoiceHyperparam(["poly3", "trig"])
        self.lasso_alpha = FloatRangeHyperparam([0.0, 100.0])

    def _transform_state(self, state):
        basis = [lambda x: x]
        if "poly3" in self.basis_functions.value:
            basis += [lambda x: x**2, lambda x: x**3]
        if "trig" in self.basis_functions.value:
            basis += [np.sin, np.cos, np.tan]
        return np.array([b(x) for b in basis for x in state])

    def traj_to_state(self, traj):
        return self._transform_state(traj[-1].obs[:])
    
    def update_state(self, state, new_obs, new_ctrl):
        return self._transform_state(new_obs)

    @property
    def state_dim(self):
        basis = [lambda x: x]
        if "poly3" in self.basis_functions.value:
            basis += [lambda x: x**2, lambda x: x**3]
        if "trig" in self.basis_functions.value:
            basis += [np.sin, np.cos, np.tan]
        return len(basis) * self.system.obs_dim

    def train(self, trajs):
        X = np.concatenate([np.apply_along_axis(self._transform_state, 1, 
            traj.obs[:-1,:]) for traj in trajs]).T
        Y = np.concatenate([np.apply_along_axis(self._transform_state, 1, 
            traj.obs[1:,:]) for traj in trajs]).T
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs]).T
        
        n = X.shape[0] # state dimension
        m = U.shape[0] # control dimension    
        
        XU = np.concatenate((X, U), axis = 0) # stack X and U together
        if self.method.value == "lstsq": # Least Squares Solution
            AB = np.dot(Y, sla.pinv2(XU))
            A = AB[:n, :n]
            B = AB[:n, n:]
        elif self.method.value == "lasso":  # Call lasso regression on coefficients
            print("Call Lasso")
            clf = Lasso(alpha=self.lasso_alpha.value)
            clf.fit(XU.T, Y.T)
            AB = clf.coef_
            A = AB[:n, :n]
            B = AB[:n, n:]
        elif self.method.value == 3: # Compute stable A, and B
            print("Compute Stable Koopman")
            # call function

        self.A, self.B = A, B

    def pred(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl
        return xpred

    def pred_diff(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl

        return xpred, np.copy(self.A)

    def to_linear(self):
        return np.copy(self.A), np.copy(self.B)

    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
