from collections import Iterable
from ..controller import Controller
from ..constraint import Constraint
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
    def __init__(self, model, cost, constraints, horizon):
        # I prefer type checking, but clearly current API does not allow me so
        assert isinstance(constraints, MPCConstraints), "constraints has to be in type autompc.control.MPCConstraints"
        self.A, self.B = model.get_linear_system()
        self.qrnf = cost.get_quadratic()
        self.constr = constraints
        self.horizon = horizon

    def _build_problem(self):
        """Use cvxpy to construct the problem"""
        nx, nu = self.B.shape
        self._x0 = cp.Parameter(nx)
        xs = cp.Variable((self.horizon, nx + 1))  # x0 is row 1
        us = cp.Variable((self.horizon, nu))
        self._xs = xs
        self._us = us
        # construct constraints
        # initial state
        cons = [xs[0] == self._x0]
        # dynamics
        for i in range(self.horizon):
            cons.append(xs[i + 1] == self.A.dot(xs[i])) + self.B.dot(us[i])
        # constraints
        if self.constr.terminal is not None:
            if self.constr.terminal.is_equality:
                cons.append(self.constr.terminal.A.dot(xs[-1]) == self.constr.terminal.l)
            else:
                tmp = self.constr.terminal.A.dot(xs[-1])
                cons.extend([tmp >= self.constr.terminal.l, tmp <= self.constr.terminal.u])
        for path in self.constr.path:
            if path.is_equality:
                for i in range(self.horizon):
                    cons.append(path.A.dot(xs[i]) + path.B.dot(us[i]) == path.l)
            else:
                for i in range(self.horizon):
                    tmp = path.A.dot(xs[i]) + path.B.dot(us[i])
                    cons.append(tmp <= path.u)
                    cons.append(tmp >= tmp.l)
        # construct the cost function from lqr cost
        obj = 0
        Q, R, N, F = self.qrnf
        obj += cp.quad_form(xs[-1], F)
        is_n_zero = np.all(N == 0)
        for i in range(self.horizon):
            obj += cp.quad_form(xs[i], Q) + cp.quad_form(us[i], R)
            if not is_n_zero:
                obj += 2 * xs[i].dot(N.dot(us[i]))
        self._problem = cp.Problem(cp.Minimize(obj))

    def _update_problem_and_solve(self, x0):
        """Solve the problem"""
        self._x0.value = x0
        self._problem.solve()
        return {'x': self._xs.value, 'u': self._us.value, 'solved': self._problem.status == 'optimal'}

    def __call__(self, x):
        rst = self._update_problem_and_solve(x)
        return rst['u'][0]  # return the first control...
