# Created by William Edwards (wre2@illinois.edu)

# External library includes
import numpy as np
import numpy.linalg as la

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (UniformIntegerHyperparameter, 
        CategoricalHyperparameter)
import ConfigSpace.conditions as CSC

# Internal library includes
from .optimizer import Optimizer
from ..utils.cs_utils import *

def _dynamic_ricatti_equation(A, B, Q, R, N, Pk):
    return (A.T @ Pk @ A 
            - (A.T @ Pk @ B + N) 
              @ la.inv(R + B.T @ Pk @ B)
              @ (B.T @ Pk @ A + N.T)
            + Q)

def _inf_horz_dt_lqr(A, B, Q, R, N, threshold=1e-3, max_iters=1000):
    P1 = Q
    P2 = _dynamic_ricatti_equation(A, B, Q, R, N, P1)
    Pdiff = np.abs(P1 - P2)
    iters = 0
    while Pdiff.max() > threshold:
        P1 = P2
        P2 = _dynamic_ricatti_equation(A, B, Q, R, N, P1)
        Pdiff = np.abs(P1 - P2)
        iters += 1
        if iters > max_iters:
            print("Warning: LQR infinite horizon failed to converge")
            break

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

class LQR(Optimizer):
    """
    Linear Quadratic Regulator (LQR) is some classical results from linear system theory and optimal control theory.
    It applies to linear system with quadratic cost function with respect to both state and control.
    It is proven that the optimal control policy is linear, i.e. :math:`u=-Kx` where :math:`x` is system state, :math:`K` is gain matrix, and :math:`u` is the control.
    The feedback :math:`K` is computed by solving Ricatti equations.
    For more details refer to these slides_ .

    .. _slides: https://katefvision.github.io/katefSlides/RECITATIONtrajectoryoptimization_katef.pdf
 
    Hyperparameters:
    
    - **finite_horizon** *(Type: str, Choices: ["true", "false"], Default: "true")*: Whether horizon is finite or infinite.
    - **horizon** *(Type: int, Low: 1, High: 1000, Default: 10)*: Length of control horizon. (Conditioned on finite_horizon="true").
    """
    def __init__(self, system):
        super().__init__(system, "LQR")

    def get_default_config_space(self):
        cs = ConfigurationSpace()
        finite_horizon = CategoricalHyperparameter(name="finite_horizon",
                choices=["true", "false"], default_value="true")
        horizon = UniformIntegerHyperparameter(name="horizon",
                lower=1, upper=1000, default_value=10)
        use_horizon = CSC.InCondition(child=horizon, parent=finite_horizon,
                values=["true"])
        cs.add_hyperparameters([horizon, finite_horizon])
        cs.add_condition(use_horizon)
        return cs

    def set_config(self, config):
        self.finite_horizon = get_hyper_bool(config, "finite_horizon")
        self.horizon = get_hyper_int(config, "horizon")

    def is_compatible(self, model, ocp):
        return (model.is_linear 
                and ocp.get_cost().is_quad
                and not ocp.are_obs_bounded
                )

    def reset(self):
        A, B = self.model.to_linear()
        state_dim = self.model.state_dim
        Q, R, F = self.ocp.get_cost().get_cost_matrices()
        Qp = np.zeros((state_dim, state_dim))
        Qp[:Q.shape[0], :Q.shape[1]] = Q
        Fp = np.zeros((state_dim, state_dim))
        Fp[:F.shape[0], :F.shape[1]] = F
        N = np.zeros((state_dim, self.system.ctrl_dim))
        self.Qp, self.Rp = Qp, R
        if self.finite_horizon:
            self.K = _finite_horz_dt_lqr(A, B, Qp, R, N, Fp, self.horizon)
        else:
            self.K = _inf_horz_dt_lqr(A, B, Qp, R, N)
        self.umin = self.ocp.get_ctrl_bounds()[:,0]
        self.umax = self.ocp.get_ctrl_bounds()[:,1]
        x0 = self.ocp.get_cost().get_goal()
        self.state0 = np.zeros(state_dim)
        self.state0[:x0.size] = x0

    def step(self, state):
        u = self.K @ (state - self.state0)
        u = np.clip(u, self.umin, self.umax)
        return u

    def get_state(self):
        return dict()

    def set_state(self, state):
        pass