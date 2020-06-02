import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace

from ..model import Model
from ..hyper import ChoiceHyperparam, MultiChoiceHyperparam

class Koopman(Model):
    def __init__(self, system):
        super().__init__(system)
        self.method = ChoiceHyperparam(["lstsq", "lasso", "stableAB"])

        self.basis_functions = MultiChoiceHyperparam(["poly3", "trig"])

    def _transform_state(self, state):
        basis = [lambda x: x]
        if "poly3" in self.basis_functions.value:
            basis += [lambda x: x**2, lambda x: x**3]
        if "trig" in self.basis_functions.value:
            basis += [np.sin, np.cos, np.tan]
        return np.array([b(x) for b in basis for x in state])

    def _state_size(self):
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
        
        # Evaluate basis functions based on states
        basis_functions = []
        
        if self.method.value == "lstsq": # Least Squares Solution
            XU = np.concatenate((X, U), axis = 0) # stack X and U together
            AB = np.dot(Y, sla.pinv2(XU))
            A = AB[:n, :n]
            B = AB[:n, n:]
        elif self.method.value == 2:  # Call lasso regression on coefficients
            print("Call Lasso")
            # body text
        elif self.method.value == 3: # Compute stable A, and B
            print("Compute Stable Koopman")
            # call function

        self.A, self.B = A, B

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
