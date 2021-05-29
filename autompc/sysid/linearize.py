
import numpy as np
from ..model import Model


class LinearizedModel(Model):
    def __init__(self, system, x0, nonlinear_model):
        super().__init__(system)
        self.x0 = x0
        #self._model = NonlinearModel(system, **model_args)
        self._model = nonlinear_model
        _, self.A, self.B = nonlinear_model.pred_diff(x0,  np.zeros(system.ctrl_dim))

    @property
    def state_dim(self):
        return self._model.state_dim

    @property
    def state_dim(self):
        return self._model.state_dim

    @staticmethod
    def get_configuration_space(system):
        raise NotImplementedError

    def traj_to_state(self, traj):
        return self._model.traj_to_state(traj)

    def update_state(self, state, new_ctrl, new_obs):
        return np.copy(new_obs)

    def to_linear(self):
        return np.copy(self.A), np.copy(self.B)

    def pred(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl

    def pred_diff(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl

    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
