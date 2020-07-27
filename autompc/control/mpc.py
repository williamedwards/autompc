from collections import Iterable
from ..controller import Controller
from ..constraint import Constraint
from ..hyper import IntRangeHyperparam
from ..cost import Cost

import cvxpy as cp
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


class LinearMPC(Controller):
    """
    Implementation of the linear controller. For this very basic version, it accepts some linear models and compute output.
    """
    def __init__(self, system, model, task, horizon=8):
        Controller.__init__(self, system, task, model)
        assert task.is_cost_quad()
        assert task.are_ineq_cons_affine()
        assert task.are_eq_cons_convex()
        self.A, self.B = model.to_linear()
        self.qrnf = task.get_quad_cost()
        try:
            self.eq_cons = task.get_affine_eq_cons()  # this is path constraint
        except:
            self.eq_cons = []
        try:
            self.ineq_cons = task.get_affine_ineq_cons()  # this is path constraint
        except:
            self.ineq_cons = []
        self.obs_bound = task.get_obs_bounds()  # dim by 2
        self.ctrl_bound = task.get_ctrl_bounds()  # dim by 2
        self.horizon = horizon
        self._built = False

    @property
    def state_dim(self):
        return self.model.state_dim

    @staticmethod
    def get_configuration_space(system, task, model):
        cs = ConfigurationSpace()
        horizon = UniformIntegerHyperparameter(name="horizon",
                lower=1, upper=100, default_value=10)
        cs.add_hyperparameter(horizon)
        return cs

    @staticmethod
    def is_compatible(system, task, model):
        #TODO: this part is really confusing...
        return (model.is_linear 
                and task.is_cost_quad()
        )
 
    def traj_to_state(self, traj):
        return self.model.traj_to_state(traj)

    def _build_problem(self):
        """Use cvxpy to construct the problem"""
        self._built = True
        nx, nu = self.B.shape
        dims, dimu = self.system.obs_dim, self.system.ctrl_dim
        self._x0 = cp.Parameter(nx)
        horizon = self.horizon
        xs = cp.Variable((horizon + 1, nx))  # x0 is row 1
        us = cp.Variable((horizon, nu))
        self._xs = xs
        self._us = us
        # construct constraints
        # initial state
        cons = [xs[0] == self._x0]
        # dynamics
        for i in range(horizon):
            cons.append(xs[i + 1] == self.A * xs[i] + self.B * us[i])
        # constraints, eq first
        if self.eq_cons:
            A, b = self.eq_cons
            for i in range(hoziron):
                cons.append(A.dot(xs[i + 1][:dims] == b))
        # ineqs
        if self.ineq_cons:
            A, b = self.ineq_cons
            for i in range(hoziron):
                cons.append(A.dot(xs[i + 1][:dims] <= b))
        # set bounds...  assume broadcasting...
        oneh = np.ones((self.horizon + 1, 1))
        cons.append(xs[:, :dims] <= oneh * self.obs_bound[:, 1])
        cons.append(xs[:, :dims] >= oneh * self.obs_bound[:, 0])
        cons.append(us[:, :dimu] <= oneh[:-1] * self.ctrl_bound[:, 1][None])
        cons.append(us[:, :dimu] >= oneh[:-1] * self.ctrl_bound[:, 0][None])
        # construct the cost function from lqr cost
        obj = 0
        Q, R, F = self.qrnf
        Q += 1e-4 * np.eye(Q.shape[0])
        obj += cp.quad_form(xs[-1][:dims], F)
        for i in range(horizon):
            obj += cp.quad_form(xs[i][:dims], Q) + cp.quad_form(us[i][:dimu], R)
        self._problem = cp.Problem(cp.Minimize(obj), cons)

    def _update_problem_and_solve(self, x0):
        """Solve the problem"""
        if not self._built:
            self._build_problem()
        self._x0.value = x0
        self._problem.solve()
        return {'x': self._xs.value, 'u': self._us.value, 'solved': self._problem.status == 'optimal'}

    def run(self, traj, latent=None):
        x = self.model.traj_to_state(traj)
        rst = self._update_problem_and_solve(x)
        return rst['u'][0][:self.system.ctrl_dim], None  # return the first control... and no hidden variable
