# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

from ..controller import Controller
from ..hyper import IntRangeHyperparam
from pdb import set_trace

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

def _finite_horz_dt_lqr(A, B, Q, R, N, horizon):
    P1 = Q
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
    def __init__(self, system, model, Q, R):
        if not model.is_linear:
            raise ValueError("Linear model required.")
        super().__init__(system, model)
        A, B, state_func, cost_func = model.to_linear()
        Qp, Rp = cost_func(Q, R)
        N = np.zeros((A.shape[0], B.shape[1]))
        self.horizon = IntRangeHyperparam((1, 100), default_value=10)
        self.K = _finite_horz_dt_lqr(A, B, Qp, Rp, N, 10)
        self.state_func = state_func
        self.Qp, self.Rp = Qp, Rp

    def run(self, traj, latent=None):
        # Implement control logic here
        state = self.state_func(traj)
        u = self.K @ state
        print("state={}".format(state))
        print("u={}".format(u))
        print("state_cost={}".format(state.T @ self.Qp @ state))

        return u, None
