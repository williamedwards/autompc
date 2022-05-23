# Created by William Edwards (wre2@illinois.edu)

import warnings

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
    Linear Quadratic Regulator (LQR) is a classical result from linear system
    theory and optimal control theory.

    It applies to linear system with quadratic cost function with respect to
    both state and control. In this setting, the optimal control policy is
    linear, i.e. :math:`u=-Kx` where :math:`x` is system state, :math:`K` is
    the gain matrix, and :math:`u` is the control.

    x' = Ax + Bu + c

    The feedback :math:`K` is computed by solving Ricatti equations.
    For more details refer to these slides_ .

    .. _slides: https://katefvision.github.io/katefSlides/RECITATIONtrajectoryoptimization_katef.pdf
 
    Hyperparameters:
    
    - **finite_horizon** *(Type: str, Choices: ["true", "false"], Default: "true")*: Whether horizon is finite or infinite.
    - **horizon** *(Type: int, Low: 1, High: 1000, Default: 10)*: Length of control horizon. (Conditioned on finite_horizon="true").
    """
    def __init__(self, system):
        super().__init__(system, "LQR")
        self.Q,self.R,self.F=None,None,None
        self.K = None
        self.state0 = None

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

    def model_requirements(self):
        return {'is_linear':True}
    
    def ocp_requirements(self):
        return {'are_obs_bounded':False}

    def cost_requirements(self):
        return {'is_quad':True}

    def reset(self):
        self.K = None

    def _recompute_gains(self):
        A, B, c = self.model.to_linear()
        if c is not None and not np.allclose(c,np.zeros(len(c))):
            warnings.warn("Linear system has a nonzero drift term, LQR control will not be optimal")
        state_dim = self.model.state_dim
        Q, R, F = self.ocp.get_cost().get_cost_matrices()
        self.Q,self.R,self.F = Q,R,F
        Qp = np.zeros((state_dim, state_dim))
        Qp[:Q.shape[0], :Q.shape[1]] = Q
        Fp = np.zeros((state_dim, state_dim))
        Fp[:F.shape[0], :F.shape[1]] = F
        N = np.zeros((state_dim, self.system.ctrl_dim))
        if self.finite_horizon:
            self.K = _finite_horz_dt_lqr(A, B, Qp, R, N, Fp, self.horizon)
        else:
            self.K = _inf_horz_dt_lqr(A, B, Qp, R, N)
        
    def set_ocp(self, ocp) -> None:
        super().set_ocp(ocp)
        Q, R, F = self.ocp.get_cost().get_cost_matrices()
        if Q is not self.Q or R is not self.R or F is not self.F:
            self.K = None  #mark that K should be recomputed
            self.Q,self.R,self.F = Q,R,F
        self.umin = self.ocp.get_ctrl_bounds()[:,0]
        self.umax = self.ocp.get_ctrl_bounds()[:,1]
        x0 = self.ocp.get_cost().goal
        self.state0 = np.zeros(self.model.state_dim)
        if x0 is not None:
            self.state0[:len(x0)] = x0

    def step(self, state):
        if self.K is None:
            self._recompute_gains()
            assert self.K is not None
        u = self.K @ (state - self.state0)
        u = np.clip(u, self.umin, self.umax)
        return u

    def get_state(self):
        return dict()

    def set_state(self, state):
        pass