# Created by William Edwards (wre2@illinois.edu)

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from ..controller import Controller

def _dynamic_ricatti_equation(A, B, Q, R, N, Pk):
    return (A.T @ Pk @ A 
            - (A.T @ Pk @ B + N) 
              @ la.inv(R + B.T @ Pk @ B)
              @ (B.T @ Pk @ A + N.T)
            + Q)

def _inf_horz_dt_lqr(A, B, Q, R, N, threshold=1e-3):
    P1 = Q
    P2 = _dynamic_ricatti_equation(A, B, Q, R, N, P1)
    Pdiff = np.abs(P1 - P2)
    while Pdiff.max() > threshold:
        P1 = P2
        P2 = _dynamic_ricatti_equation(A, B, Q, R, N, P1)
        Pdiff = np.abs(P1 - P2)

    K = -la.inv(R + B.T @ P2 @ B) @ B.T @ P2 @ A

    return K

def _finite_horz_dt_lqr(A, B, Q, R, N, F, horizon):
    P1 = F
    P2 = _dynamic_ricatti_equation(A, B, Q, R, N, P1)
    for _ in range(horizon):
        P1 = P2
        P2 = _dynamic_ricatti_equation(A, B, Q, R, N, P1)
        Pdiff = np.abs(P1 - P2)

    K = -la.inv(R + B.T @ P2 @ B) @ B.T @ P2 @ A

    return K

class InfiniteHorizonLQR(Controller):
    def __init__(self, system, model, Q, R):
        if not model.is_linear:
            raise ValueError("Linear model required.")
        super().__init__(system, model)
        A, B, state_func, cost_func = model.to_linear()
        Qp, Rp = cost_func(Q, R)
        N = np.zeros((A.shape[0], B.shape[1]))
        self.K = _inf_horz_dt_lqr(A, B, Qp, Rp, N)
        self.state_func = state_func
        self.Qp, self.Rp = Qp, Rp

    def run(self, traj, latent=None):
        # Implement control logic here
        u = self.K @ self.state_func(traj)

        return u, None

class FiniteHorizonLQR(Controller):
    def __init__(self, system, task, model, horizon):
        super().__init__(system, task, model)
        A, B = model.to_linear()
        N = np.zeros((A.shape[0], B.shape[1]))
        self.horizon = horizon
        state_dim = model.state_dim
        Q, R, F = task.get_quad_cost()
        Qp = np.zeros((state_dim, state_dim))
        Qp[:Q.shape[0], :Q.shape[1]] = Q
        Fp = np.zeros((state_dim, state_dim))
        Fp[:F.shape[0], :F.shape[1]] = F
        self.K = _finite_horz_dt_lqr(A, B, Qp, R, N, Fp, horizon)
        self.Qp, self.Rp = Qp, R
        self.model = model

    @property
    def state_dim(self):
        return self.model.state_dim + self.system.ctrl_dim

    @staticmethod
    def get_configuration_space(system, task, model):
        cs = ConfigurationSpace()
        horizon = UniformIntegerHyperparameter(name="horizon",
                lower=1, upper=100, default_value=10)
        cs.add_hyperparameter(horizon)
        return cs

    @staticmethod
    def is_compatible(system, task, model):
        return (model.is_linear 
                and task.is_cost_quad()
                and not task.are_obs_bounded()
                and not task.are_ctrl_bounded()
                and not task.eq_cons_present()
                and not task.ineq_cons_present())
 
    def traj_to_state(self, traj):
        return np.concatenate([self.model.traj_to_state(traj),
                traj[-1].ctrl])

    def run(self, state, new_obs):
        # Implement control logic here
        modelstate = self.model.update_state(state[:-self.system.ctrl_dim],
                state[-self.system.ctrl_dim:], new_obs)
        u = self.K @ modelstate
        print("state={}".format(state))
        print("u={}".format(u))
        print("state_cost={}".format(modelstate.T @ self.Qp @ modelstate))
        statenew = np.concatenate([modelstate, u])

        return u, statenew
