import numpy as np
from pdb import set_trace

from .model import Model
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

class DummyLinear(Model):
    def __init__(self, system, A, B):
        super().__init__(system)
        self.A = A
        self.B = B

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        return cs

    def traj_to_state(self, traj):
        return traj[-1].obs[:]
    
    def update_state(self, state, new_ctrl, new_obs):
        return np.copy(new_obs)

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs):
        pass

    def pred(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl
        return xpred

    def pred_diff(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl

        return xpred, np.copy(self.A), np.copy(self.B)

    def to_linear(self):
        return np.copy(self.A), np.copy(self.B)

    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
