# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from ..model import Model
from ..hyper import ChoiceHyperparam, MultiChoiceHyperparam

class Koopman(Model):
    def __init__(self, system):
        super().__init__(system)
        self.method = ChoiceHyperparam(["ridge", "lasso", "stableAB"])

        #TODO Consider in more detail
        self.basis_functions = MultiChoiceHyperparam(["sin", "cos", "tan",
            "sinh", "cosh", "sq", "cube", "sqrt"])

    def train(self, trajs):
        Measurements = np.concatenate([traj.obs for traj in trajs])
        X = Measurements[:, 0:-1]
        Y = Measurements[:, 1:]
        U = np.concatenate([traj.ctrls for traj in trajs])
        
        n = X.shape[0] # state dimension
        m = U.shape[0] # control dimension    
        
        # Evaluate basis functions based on states
        
        if self.method.value == 1: # Least Squares Solution
            XU = np.concatenate((X, U), axis = 0) # stack X and U together
            AB = dot(Y, linalg.pinv2(XU))
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

        xnew = self.A @ x + self.B @ u

        # Transform to original state space xpred

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
        return np.copy(self.A), np.copy(self.B), state_func, cost_func

    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])


