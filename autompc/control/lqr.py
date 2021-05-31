# Created by William Edwards (wre2@illinois.edu)

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (UniformIntegerHyperparameter, 
        CategoricalHyperparameter)
import ConfigSpace.conditions as CSC

from .controller import Controller, ControllerFactory

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
    print("P2=")
    print(P2)

    return K

#class InfiniteHorizonLQR(Controller):
#    def __init__(self, system, model, Q, R):
#        if not model.is_linear:
#            raise ValueError("Linear model required.")
#        super().__init__(system, model)
#        A, B, state_func, cost_func = model.to_linear()
#        Qp, Rp = cost_func(Q, R)
#        N = np.zeros((A.shape[0], B.shape[1]))
#        self.K = _inf_horz_dt_lqr(A, B, Qp, Rp, N)
#        self.state_func = state_func
#        self.Qp, self.Rp = Qp, Rp
#
#    def run(self, traj, latent=None):
#        # Implement control logic here
#        u = self.K @ self.state_func(traj)
#
#        return u, None

# class LQRFactory(ControllerFactory):
#     """
#     Docs
# 
#     Hyperparameters:
#     
#     - *horizon_type* (Type: str, Choices: ["finite", "infinite"], Default: "finite"): Whether horizon is finite or infinite.
#     - *horizon* (Type: int, Low: 1, High: 1000, Default: 10): Length of control horizon. (Conditioned on horizon_type="finite").
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(self, *args, **kwargs)
#         self.name  = "InfiniteHorizonLQR"
# 
#     def get_configuration_space(self):
#         cs = ConfigurationSpace()
#         horizon_type = CategoricalHyperparameter("horizon_type", choices=["finite", "infinite"], default="finite")
#         horizon = UniformIntegerHyperparameter(name="horizon_cond",
#                 lower=1, upper=1000, default_value=10)
#         horizon_cond = InCondition(child=horizon, parent=horizon_type, values=["finite"])
#         cs.add_hyperparameters([horizon, horizon_type])
#         cs.add_condition(horizon_cond)
#         return cs
# 
#     def __call__(self, cfg, task, model):
#         if cfg["horizon_type"] == "finite":
#             controller = FiniteHorizonLQR(self.system, task, model, horizon = cfg["horizon"])
#         else:
#             controller = InfiniteHorizonLQR(self.system, task, model)

class InfiniteHorizonLQR(Controller):
    def __init__(self, system, task, model):
        super().__init__(system, task, model)
        A, B = model.to_linear()
        state_dim = model.state_dim
        Q, R, F = task.get_cost().get_cost_matrices()
        Qp = np.zeros((state_dim, state_dim))
        Qp[:Q.shape[0], :Q.shape[1]] = Q
        X, L, K = dare(A, B, Qp, R)
        self.K = K
        self.Qp, self.Rp = Qp, R
        self.model = model

    @property
    def state_dim(self):
        return self.model.state_dim + self.system.ctrl_dim

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
        u = np.array(self.K @ modelstate).flatten()
        print("state={}".format(state))
        print("u={}".format(u))
        print("state_cost={}".format(modelstate.T @ self.Qp @ modelstate))
        statenew = np.concatenate([modelstate, u])

        return u, statenew

class FiniteHorizonLQR(Controller):
    def __init__(self, system, task, model, horizon):
        super().__init__(system, task, model)
        A, B = model.to_linear()
        N = np.zeros((A.shape[0], B.shape[1]))
        self.horizon = horizon
        state_dim = model.state_dim
        #Q, R, F = task.get_quad_cost()
        Q, R, F = task.get_cost().get_cost_matrices()
        Qp = np.zeros((state_dim, state_dim))
        Qp[:Q.shape[0], :Q.shape[1]] = Q
        Fp = np.zeros((state_dim, state_dim))
        Fp[:F.shape[0], :F.shape[1]] = F
        self.K = _finite_horz_dt_lqr(A, B, Qp, R, N, Fp, horizon)
        self.Qp, self.Rp = Qp, R
        self.model = model
        self.umin = task.get_ctrl_bounds()[:,0]
        self.umax = task.get_ctrl_bounds()[:,1]

    @property
    def state_dim(self):
        return self.model.state_dim + self.system.ctrl_dim

    @staticmethod
    def is_compatible(system, task, model):
        return (model.is_linear 
                and task.is_cost_quad()
                and not task.are_obs_bounded()
                #and not task.are_ctrl_bounded()
                and not task.eq_cons_present()
                and not task.ineq_cons_present())
 
    def traj_to_state(self, traj):
        return np.concatenate([self.model.traj_to_state(traj),
                traj[-1].ctrl])

    def run(self, state, new_obs):
        # Implement control logic here
        modelstate = self.model.update_state(state[:-self.system.ctrl_dim],
                state[-self.system.ctrl_dim:], new_obs)
        x0 = self.task.get_cost().get_x0()
        if x0.size < modelstate.size:
            state0 = np.zeros(modelstate.size)
            state0[:x0.size] = x0
        else:
            state0 = x0
        u = self.K @ (modelstate - state0)
        u = np.minimum(u, self.umax)
        u = np.maximum(u, self.umin)
        #print("state={}".format(state))
        #print("u={}".format(u))
        #print("state_cost={}".format(modelstate.T @ self.Qp @ modelstate))
        statenew = np.concatenate([modelstate, u])

        return u, statenew

class LQRFactory(ControllerFactory):
    """
    Linear Quadratic Regulator (LQR) is some classical results from linear system theory and optimal control theory.
    It applies to linear system with quadratic cost function with respect to both state and control.
    It is proven that the optimal control policy is linear, i.e. :math:`u=-Kx` where :math:`x` is system state, :math:`K` is gain matrix, and :math:`u` is the control.
    The feedback :math:`K` is computed by solving Ricatti equations.
    For more details we refer to `this slide <https://katefvision.github.io/katefSlides/RECITATIONtrajectoryoptimization_katef.pdf>`.
 
    Hyperparameters:
    
    - *finite_horizon* (Type: str, Choices: ["true", "false"], Default: "finite"): Whether horizon is finite or infinite.
    - *horizon* (Type: int, Low: 1, High: 1000, Default: 10): Length of control horizon. (Conditioned on finite_horizon="true").
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.Controller = LQR
        self.name = "LQR"

    def get_configuration_space(self):
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

class LQR(Controller):
    def __init__(self, system, task, model, finite_horizon, horizon=None):
        if not isinstance(finite_horizon, bool):
            finite_horizon = True if finite_horizon == "true" else False
        if finite_horizon:
            self._controller = FiniteHorizonLQR(system, task, model, horizon)
        else:
            self._controller = InfiniteHorizonLQR(system, task, model)

    @property
    def state_dim(self):
        return self._controller.state_dim

    @staticmethod
    def is_compatible(system, task, model):
        return (model.is_linear 
                and task.is_cost_quad()
                and not task.are_obs_bounded()
                #and not task.are_ctrl_bounded()
                and not task.eq_cons_present()
                and not task.ineq_cons_present())

    def traj_to_state(self, traj):
        return self._controller.traj_to_state(traj)

    def run(self, state, new_obs):
        return self._controller.run(state, new_obs)
