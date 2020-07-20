from collections import Iterable
from ..controller import Controller
from ..constraint import Constraint
from ..hyper import IntRangeHyperparam
from ..cost import Cost

import cvxpy as cp
import numpy as np


class LinearConstraint(Constraint):
    r"""Define a linear constraint of type

    .. math::
        l \le A x + B u \le u

    Parameters
    ----------
        A : (Numpy array)
            the linear matrix associated with states
        B : (Numpy array)
            the linear matrix associated with control
        l : (Numpy array)
            lower bound for the constraints
        u : (Numpy array or None)
            upper bound for the constraints or None if equality constraint
    """
    def __init__(self, A, B, l, u):
        self.A = A
        self.B = B
        self.l = l
        if u is None:
            self.u = l
            self.is_equality = True
        else:
            self.u = u
            self.is_equality = np.allclose(l, u)

    def eval(self, xs, us, ret_grad=False):
        """
        Parameters
        ----------
            xs : Numpy array
                Trajectory states
            us : Numpy array
                Trajectory controls
            ret_grad : bool
                If true, return gradients of numeric constraints,
                if available (otherwise raise NotImplementedErorr).
        Returns 
        -------
            cons_vals : Numpy array
                Must be nonnegative to satisfy.
            cons_grad : Optional, Numpy array
                Gradient of cons_vals wrt xs and us.
        May not be implemented for all constraints.
        """
        val = self.A.dot(xs) + self.B.dot(us)
        return np.concatenate((val - self.l), (self.u - val))


class LQRCost(Cost):
    def __init__(self, Q, R, F=None):
        self.Q = Q
        self.R = R
        if F is None:
            self.F = np.zeros_like(Q)
        else:
            self.F = F

    def get_quadratic(self):
        return self.Q, self.R, None, self.F
    
    def get_terminal(self, x, ret_grad=False):
        cost = x.dot(self.F.dot(x))
        if ret_grad:
            grad = 2 * self.F.dot(x)
            return cost, grad
        else:
            return cost

    def get_additive(self, x, u, ret_grad=False):
        costx = x.dot(self.Q.dot(x))
        costu = u.dot(self.R.dot(u))
        if ret_grad:
            g1 = 2 * self.Q.dot(x)
            g2 = 2 * self.R.dot(u)
            return costx + costu, g1, g2
        else:
            return costx


class MPCConstraints(object):
    """This class maintains both path and terminal constraints for an MPC problem.
    
    Parameters
    ----------
        path: LinearConstraint or array-like of LinearConstraint
            This variable specifies all the path constraints we have to satisfy
        terminal: LinearConstraint
            Just the terminal constraint. At this moment, we do not need control B so a check is necessary
    """
    def __init__(self, path, terminal):
        assert terminal is None or (isinstance(terminal, LinearConstraint) and np.all(terminal.B == 0)), "terminal constraint incorrectly defined"
        if isinstance(path, Iterable):
            self.path = path
        else:
            self.path = [path]
        self.terminal = terminal


class LinearMPC(Controller):
    """
    Implementation of the linear controller. For this very basic version, it accepts some linear models and compute output.
    """
    def __init__(self, system, task, model):
        Controller.__init__(self, system, model)
        assert task.is_cost_quad()
        assert task.are_ineq_cons_affine()
        assert task.are_eq_cons_convex()
        self.A, self.B, self.state_func, self.cfun = model.to_linear()
        self.qrnf = task.get_quad_cost()
        self.eq_cons = task.get_affine_eq_cons()  # this is path constraint
        self.ineq_cons = task.get_affine_ineq_cons()  # this is path constraint
        self.obs_bound = task.get_obs_bounds()  # dim by 2
        self.ctrl_bound = task.get_ctrl_bounds()  # dim by 2
        self.horizon = IntRangeHyperparam((5, 20), default_value=8)
        self._built = False

    def _build_problem(self):
        """Use cvxpy to construct the problem"""
        self._built = True
        nx, nu = self.B.shape
        self._x0 = cp.Parameter(nx)
        horizon = self.horizon.value
        xs = cp.Variable((horizon + 1, nx))  # x0 is row 1
        us = cp.Variable((horizon, nu))
        self._xs = xs
        self._us = us
        # construct constraints
        # initial state
        cons = [xs[0] == self._x0]
        # dynamics
        for i in range(horizon):
            cons.append(xs[i + 1] == self.A * (xs[i]) + self.B * (us[i]))
        # constraints, eq first
        A, b = self.eq_cons
        if A.shape[0] > 0:
            for i in range(hoziron):
                cons.append(A.dot(xs[i + 1] == b))
        # ineqs
        A, b = self.ineq_cons
        if A.shape[0] > 0:
            for i in range(hoziron):
                cons.append(A.dot(xs[i + 1] <= b))
        # set bounds...  assume broadcasting...
        cons.append(xs <= self.obs_bound[:, 1])
        cons.append(xs >= self.obs_bound[:, 0])
        cons.append(us <= self.ctrl_bound[:, 1])
        cons.append(us >= self.ctrl_bound[:, 0])
        # construct the cost function from lqr cost
        obj = 0
        Q, R, F = self.qrnf
        Q, R, F = self.cfun(Q, R, F)  # TODO: clarify if we should just ignore N, we probably should
        Q += 1e-4 * np.eye(Q.shape[0])
        obj += cp.quad_form(xs[-1], F)
        for i in range(horizon):
            obj += cp.quad_form(xs[i], Q) + cp.quad_form(us[i], R)
        self._problem = cp.Problem(cp.Minimize(obj), cons)

    def _update_problem_and_solve(self, x0):
        """Solve the problem"""
        if not self._built:
            self._build_problem()
        self._x0.value = x0
        self._problem.solve()
        return {'x': self._xs.value, 'u': self._us.value, 'solved': self._problem.status == 'optimal'}

    def run(self, traj, latent=None):
        x = self.state_func(traj)
        rst = self._update_problem_and_solve(x)
        print(rst)
        return rst['u'][0], None  # return the first control... and no hidden variable
