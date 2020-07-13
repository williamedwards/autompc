from collections import Iterable
from ..controller import Controller
from ..constraint import Constraint
from ..hyper import IntRangeHyperparam
from ..cost import Cost

import cvxpy as cp
import numpy as np

from pyoptsolver import OptProblem, OptConfig, OptSolver


class TrajOptProblem(object):
    """Just a general interface for nonlinear optimization problems.
    I will just use knitro/ipopt style and the snopt one is easily written as well.

    Args:
        nx (int): dimension of the decision variable
        nc (int): dimension of the constraints
    """
    def __init__(self, nx, nc):
        self.dimx = nx
        self.dimc = nc
        self.xlb, self.xub = np.zeros((2, nx))
        self.clb, self.cub = np.zeros((2, nc))

    def get_cost(self, x):
        raise NotImplementedError("Sub-class has to implement get_cost function.")

    def get_gradient(self, x):
        raise NotImplementedError("Sub-class has to implement get_gradient function.")

    def get_constraint(self, x):
        raise NotImplementedError("Sub-class has to implement get_constraint function.")

    def get_jacobian(self, x, return_rowcol):
        """This function computes the Jacobian at current solution x, if return_rowcol is True, it has to return row and col, too."""
        raise NotImplementedError("Sub-class has to implement get_jacobian function.")


class BoundedConstraint(object):
    """This class implements constraints that is like lb <= c <= ub"""
    def __init__(self, dim, lb, ub):
        self.dim = dim
        self.lb = lb
        self.ub = ub

    def eval(self, xs, ret_grad=False):
        raise NotImplementedError('Subclass has to implement eval function')


class PathConstraint(BoundedConstraint):
    def __init__(self, dim, lb, ub):
        BoundedConstraint.__init__(self, dim, lb, ub)

    def eval(self, x, u, ret_grad=False):
        """For path constraint, both x and u are provided. For terminal, u is None and ret_grad returns one jacobian only"""
        raise NotImplementedError("Subclass has to implement eval function.")


class ConstrContainer(object):
    """This container include information of constraints.
    Basically, the user can insert various types of constraint into this container.
    During insertion, they have to specify which point this constraint is evaluated.
    """
    def __init__(self):
        self.terminal_constrs = []
        self.point_constrs = []

    def add_terminal(self, constr):
        self.terminal_constrs.append(constr)
    
    def add_point(self, index, constr):
        assert index >= 0
        self.point_constrs.append((index, constr))

    def compute_dim(self, horizon):
        """Compute dimension of all constraints, it depends on horizon"""
        dim1 = sum([tc.dim for tc in self.terminal_constrs])
        dim2 = sum([pc.dim * (horizon if index is None else 1) for index, pc in self.point_constrs])
        return dim1 + dim2


class NonLinearMPCProblem(TrajOptProblem):
    """Just write the NonLinear MPC problem in the OptProblem style.
    """
    def __init__(self, system, cost, constr, horizon, xbd=None, ubd=None):
        assert isinstance(constr, ConstrContainer)
        self.system = system
        dc = system.system.ctrl_dim
        ds = system.system.obs_dim
        self.ctrl_dim = dc
        self.obs_dim = ds
        self.cost = cost
        self.constr = constr
        # now I can get the size of the problem
        nx = ds * (horizon + 1) + dc * (horizon)  # x0 to xN, u0 to u_{N-1}
        nf = constr.compute_dim(horizon) + horizon * ds  # for dynamics and other constraints
        self.horizon = horizon
        TrajOptProblem.__init__(self, nx, nf)
        self._create_cache()

    def _create_cache(self):
        self._x = np.zeros(self.dimx)
        self._grad = np.zeros(self.dimx)
        self._c = np.zeros(self.dimc)
        self._c_dyn = self._c[-self.horizon * self.obs_dim:].reshape((self.horizon, -1))  # the last parts store dynamics
        len1 = (self.horizon + 1) * self.obs_dim
        len2 = self.horizon * self.ctrl_dim
        self._state = self._x[:len1].reshape((self.horizon + 1, self.obs_dim))
        self._ctrl = self._x[len1:].reshape((self.horizon, self.ctrl_dim))
        self._grad_state = self._grad[:len1].reshape((self.horizon + 1, self.obs_dim))
        self._grad_ctrl = self._grad[len1:].reshape((self.horizon, self.ctrl_dim))
        self._x[:] = np.random.random(self.dimx)
        self._row, self._col = self.get_jacobian(self._x, True)
        self._jac = np.zeros(self._row.size)
    
    @property
    def nnz(self):
        return self._jac.size

    def get_cost(self, x):
        # compute the cost function, not sure how it's gonna be written though
        self._x[:] = x  # copy contents in
        tc = self.cost.get_terminal(self._state[-1])
        for i in range(self.horizon):
            tc += self.cost.get_additive(self._state[i], self._ctrl[i])
        return tc

    def get_gradient(self, x):
        """Compute the gradient given some guess"""
        self._x[:] = x
        self._grad[:] = 0  # reset just in case
        # terminal one
        _, gradtc = self.cost.get_terminal(self._state[-1], ret_grad=True)
        self._grad_state[-1] = gradtc
        for i in range(self.horizon):
            _, gradxi, gradui = self.cost.get_additive(self._state[i], self._ctrl[i], ret_grad=True)
            self._grad_state[i] = gradxi
            self._grad_ctrl[i] = gradui
        return self._grad

    def get_constraint(self, x):
        """Evaluate the constraint function"""
        self._x[:] = x
        self._c[:] = 0
        # first compute for dynamics
        for i in range(self.horizon):
            self._c_dyn[i] = -self._state[i + 1] + self.system.pred(self._state[i], self._ctrl[i])
        # then terminal constraints
        cr = 0  # means currow
        for tc in self.constr.terminal_constrs:
            self._c[cr: cr + tc.dim] = tc.eval(self._state[-1])
            cr += tc.dim
        # then other point constraints
        for idx, pc in self.constr.point_constrs:
            if idx is None:
                for i in range(self.horizon):
                    self._c[cr: cr + pc.dim] = pc.eval(self._state[i], self._ctrl[i])
                    cr += pc.dim
            else:
                self._c[cr: cr + pc.dim] = pc.eval(self._state[idx], self._ctrl[idx])
                cr += pc.dim
        return self._c

    def get_constr_bounds(self):
        """Just return the bounds of constraints"""
        clb, cub = np.zeros((2, self.dimc))
        # start from terminal_constrs
        cr = 0
        for tc in self.constr.terminal_constrs:
            crv = cr + tc.dim
            clb[cr: crv] = tc.lb
            cub[cr: crv] = tc.ub
            crv = cr
        # them point constraints
        for idx, pc in self.constr.point_constrs:
            if idx is None:
                crv = cr + pc.dim * self.horizon
                clb[cr: crv].reshape((self.horizon, pc.dim))[:] = pc.lb
                cub[cr: crv].reshape((self.horizon, pc.dim))[:] = pc.lb
            else:
                crv = cr + pc.dim
                clb[cr: crv] = pc.lb
                cub[cr: crv] = pc.ub
                cr = crv
        return clb, cub

    def get_variable_bounds(self, statebds=None, ctrlbds=None):
        if statebds is None and ctrlbds is None:
            return -1e20 * np.ones(self.dimx), 1e20 * np.ones(self.dimx)
        dc = self.ctrl_dim
        ds = self.obs_dim
        if statebds is None:
            slb = -1e20 * np.ones(ds)
            sub = -slb
        else:
            slb, sub = statebds
        if ctrlbds is None:
            ulb = -1e20 * np.ones(dc)
            uub = 1e20 * np.ones(dc)
        else:
            ubl, uub = ctrlbds
        xlb, xub = np.zeros((2, self.dimx))
        tmp1 = xlb[:self.horizon * (dc + ds)]
        tmp2 = xub[:self.horizon * (dc + ds)]
        tmp1[:, :ds] = slb
        tmp2[:, :ds] = sub
        slb[-ds:] = slb
        sub[-ds:] = sub
        tmp1[:, ds:] = ulb
        tmp2[:, ds:] = uub
        return xlb, xub

    def _dense_to_rowcol(self, shape, row0, col0):
        row, col = shape
        rows = np.arange(row)[:, None] * np.ones(col) + row0
        cols = np.ones((row, 1)) * np.arange(col) + col0
        return rows.flatten(), cols.flatten()

    def get_state_index(self, index):
        return index * self.obs_dim

    def get_ctrl_index(self, index):
        return (self.horizon + 1) * self.obs_dim + index * self.ctrl_dim

    def get_jacobian(self, x, return_rowcol):
        """This function computes the Jacobian at current solution x, if return_rowcol is True, it returns a tuple of the patterns of row and col"""
        self._x[:] = x
        # Here I may as well assume all the  ret_grad stuff returns a dense jacobian or None which means all zero, support for coo_matrix is under development
        dims = self._state.shape[1]
        dimu = self._ctrl.shape[1]
        if return_rowcol:
            cr = 0
            row = []
            col = []
            # for terminal constraints first
            tc_idx = self.get_state_index(self.horizon)
            for tc in self.constr.terminal_constrs:
                _, jac = tc.eval(self._state[-1], ret_grad=True)
                rowptn, colptn = self._dense_to_rowcol(jac.shape, cr, tc_idx)
                row.append(rowptn)
                col.append(colptn)
                cr += tc.dim
            # then other point constraints
            base_x_idx = 0
            base_u_idx = self.get_ctrl_index(0)
            for idx, pc in self.constr.point_constrs:
                _, jac1, jac2 = pc.eval(self._state[0], self._ctrl[0], ret_grad=True)
                if jac1 is not None:
                    rowjac1, coljac1 = self._dense_to_rowcol(jac1.shape, 0, 0)
                else:
                    rowjac1 = coljac1 = np.empty(0)
                if jac2 is not None:
                    rowjac2, coljac2 = self._dense_to_rowcol(jac2.shape, 0, 0)
                else:
                    rowjac2, coljac2 = np.empty(0)
                if idx is None:  # for this, I just compute one and populate elsewhere
                    for i in range(self.horizon):
                        row.append(rowjac1 + cr)
                        col.append(coljac1 + base_x_idx + i * dims)
                        row.append(rowjac2 + cr)
                        col.append(coljac2 + base_u_idx + i * dimu)
                        cr += pc.dim
                else:
                    assert idx >= 0
                    row.append(rowjac1 + cr)
                    col.append(coljac1 + base_x_idx + idx * dims)
                    row.append(rowjac2 + cr)
                    col.append(coljac2 + base_u_idx + idx * dimu)
                    cr += pc.dim
            # finally for dynamics
            _, mat1 = self.system.pred_diff(self._state[0], self._ctrl[0])
            srowptn, scolptn = self._dense_to_rowcol(mat1[:, :dims].shape, 0, 0)
            urowptn, ucolptn = self._dense_to_rowcol(mat1[:, dims:].shape, 0, 0)
            # compute patterns for it
            for i in range(self.horizon):
                row.append(cr + srowptn)
                col.append(base_x_idx + i * dims + scolptn)
                row.append(cr + urowptn)
                col.append(base_u_idx + i * dimu + ucolptn)
                # take care, here you are placing them after placing jacobian
                row.append(cr + np.arange(dims))
                col.append(base_x_idx + (i + 1) * dims + np.arange(dims))
                cr += dims
            return np.concatenate(row), np.concatenate(col)
        else:
            # I have to compute the jacobian here
            cr = 0
            cg = 0
            self._jac[:] = 0
            # for terminal constraints first
            tc_idx = self.get_state_index(self.horizon)
            for tc in self.constr.terminal_constrs:
                _, jac = tc.eval(self._state[-1], ret_grad=True)
                self._jac[cg: cg + jac.size] = jac.flat
                cg += jac.size
            # then other point constraints
            for idx, pc in self.constr.point_constrs:
                if idx is None:  # for this, I just compute one and populate elsewhere
                    for i in range(self.horizon):
                        _, jac1, jac2 = pc.eval(self._state[i], self._ctrl[i], ret_grad=True)
                        if jac1 is not None:
                            self._jac[cg: cg + jac1.size] = jac1.flat
                            cg += jac1.size
                        if jac2 is not None:
                            self._jac[cg: cg + jac2.size] = jac2.flat
                            cg += jac2.size
                else:
                    _, jac1, jac2 = pc.eval(self._state[idx], self._ctrl[idx], ret_grad=True)
                    if jac1 is not None:
                        self._jac[cg: cg + jac1.size] = jac1.flat
                        cg += jac1.size
                    if jac2 is not None:
                        self._jac[cg: cg + jac2.size] = jac2.flat
                        cg += jac2.size
            # finally for dynamics
            # compute patterns for it
            for i in range(self.horizon):
                _, mat = self.system.pred_diff(self._state[i], self._ctrl[i])
                mats = mat[:, :dims]
                matu = mat[:, dims:]
                self._jac[cg: cg + mats.size] = mats.flat
                cg += mats.size
                self._jac[cg: cg + matu.size] = matu.flat
                cg += matu.size
                self._jac[cg: cg + dims] = -1
                cg += dims
            return self._jac


class IpoptWrapper(OptProblem):
    """Just the ipopt style stuff"""
    def __init__(self, prob):
        assert isinstance(prob, TrajOptProblem)
        self.prob = prob
        OptProblem.__init__(self, prob.dimx, prob.dimc, prob.nnz)
        self.get_lb()[:], self.get_ub()[:] = prob.get_constr_bounds()
        self.get_xlb()[:], self.get_xub()[:] = prob.get_variable_bounds()
        self.ipopt_style()

    def __cost__(self, x):
        return self.prob.get_cost(x)

    def __gradient__(self, x, y):
        y[:] = self.prob.get_gradient(x)
        return True

    def __constraint__(self, x, y):
        y[:] = self.prob.get_constraint(x)
        return 0

    def __jacobian__(self, x, jac, row, col, rec):
        if rec:
            row_, col_ = self.prob.get_jacobian(x, True)
            row[:] = row_
            col[:] = col_
        else:
            jac[:] = self.prob.get_jacobian(x, False)
        return 0


class NonLinearMPC(Controller):
    """
    Implementation of the linear controller. For this very basic version, it accepts some linear models and compute output.
    constraints is a dict of constraints we have to consider, it has two keys: path and terminal. The items are list of Constraints.
    cost is a Cost instance to compute fitness of a trajectory
    """
    def __init__(self, system, model, cost, constraints=None):
        # I prefer type checking, but clearly current API does not allow me so
        Controller.__init__(self, system, model)
        self.cost_fun = cost
        self.constr = constraints if constraints is not None else ConstrContainer()
        self.horizon = IntRangeHyperparam((1, 10))
        self._built = False

    def _build_problem(self):
        """Use cvxpy to construct the problem"""
        self._built = True
        self.problem = NonLinearMPCProblem(self.system, self.cost_fun, self.constr, self.horizon.value)
        self.wrapper = IpoptWrapper(self.problem)

    def _update_problem_and_solve(self, x0):
        """Solve the problem"""
        if not self._built:
            self._build_problem()
        dims = self.problem.obs_dim
        self.wrapper.get_xlb()[:dims] = self.wrapper.get_xub()[:dims] = x0  # so I set this one
        config = OptConfig(backend='ipopt')
        solver = OptSolver(self.wrapper, config)
        rst = solver.solve_rand()
        return rst

    def run(self, traj, latent=None):
        x = self.model.traj_to_state(traj)
        rst = self._update_problem_and_solve(x)
        print(rst.flag)
        sol = rst.sol.copy()
        dims = self.problem.obs_dim
        dimu = self.problem.ctrl_dim
        idx0 = dims * (self.horizon.value + 1)
        return sol[idx0: idx0 + dimu], None
