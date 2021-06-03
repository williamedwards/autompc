from collections import Iterable
from .controller import Controller, ControllerFactory
from pdb import set_trace
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

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

class NonLinearMPCProblem(TrajOptProblem):
    """Just write the NonLinear MPC problem in the OptProblem style.
    """
    def __init__(self, system, model, task, horizon):
        self.system = system
        self.task = task
        self.model = model
        self.horizon = horizon
        dc = system.ctrl_dim
        ds = system.obs_dim
        self.ctrl_dim = dc
        self.obs_dim = ds
        # now I can get the size of the problem
        nx = ds * (horizon + 1) + dc * horizon  # x0 to xN, u0 to u_{N-1}
        nf = horizon * ds  # for dynamics and other constraints
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
        cost = self.task.get_cost()
        self._x[:] = x  # copy contents in
        dt = self.system.dt
        tc = cost.eval_term_obs_cost(self._state[-1])
        for i in range(self.horizon + 1):
            tc += cost.eval_obs_cost(self._state[i]) * dt
        for i in range(self.horizon):
            tc += cost.eval_ctrl_cost(self._ctrl[i]) * dt
        return tc

    def get_gradient(self, x):
        """Compute the gradient given some guess"""
        self._x[:] = x
        self._grad[:] = 0  # reset just in case
        # terminal one
        cost = self.task.get_cost()
        _, gradtc = cost.eval_term_obs_cost_diff(self._state[-1])
        self._grad_state[-1] = gradtc
        dt = self.system.dt
        for i in range(self.horizon + 1):
            _, gradx = cost.eval_obs_cost_diff(self._state[i])
            self._grad_state[i] += gradx * dt
        for i in range(self.horizon):
            _, gradu = cost.eval_ctrl_cost_diff(self._ctrl[i])
            self._grad_ctrl[i] = gradu * dt
        return self._grad

    def get_constraint(self, x):
        """Evaluate the constraint function"""
        self._x[:] = x
        self._c[:] = 0
        # first compute for dynamics
        pred_states = self.model.pred_batch(self._state[:self.horizon], self._ctrl[:self.horizon])
        for i in range(self.horizon):
            self._c_dyn[i] = -self._state[i + 1] + pred_states[i]
        return self._c

    def get_constr_bounds(self):
        """Just return the bounds of constraints"""
        clb, cub = np.zeros((2, self.dimc))
        return clb, cub

    def get_variable_bounds(self):
        obsbd = self.task.get_obs_bounds()
        ctrlbd = self.task.get_ctrl_bounds()
        dc = self.ctrl_dim
        ds = self.obs_dim
        xlb, xub = np.zeros((2, self.dimx))
        xlb[:(self.horizon + 1) * ds].reshape((-1, ds))[:] = obsbd[:, 0]
        xub[:(self.horizon + 1) * ds].reshape((-1, ds))[:] = obsbd[:, 1]
        xlb[-self.horizon * dc:].reshape((-1, dc))[:] = ctrlbd[:, 0]
        xub[-self.horizon * dc:].reshape((-1, dc))[:] = ctrlbd[:, 1]
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
        dims = self.obs_dim
        dimu = self.ctrl_dim
        if return_rowcol:
            cr = 0
            row = []
            col = []
            _, mat1, mat2 = self.model.pred_diff(self._state[0], self._ctrl[0])
            srowptn, scolptn = self._dense_to_rowcol(mat1.shape, 0, 0)
            urowptn, ucolptn = self._dense_to_rowcol(mat2.shape, 0, 0)
            # compute patterns for it
            base_x_idx = 0
            base_u_idx = dims * (self.horizon + 1)
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
            ###### Placeholder for terminal constraints
            # then other point constraints
            _, matss, matus = self.model.pred_diff_parallel(self._state[:self.horizon], self._ctrl[:self.horizon])
            for i in range(self.horizon):
                mats, matu = matss[i], matus[i]
                self._jac[cg: cg + mats.size] = mats.flat
                cg += mats.size
                self._jac[cg: cg + matu.size] = matu.flat
                cg += matu.size
                self._jac[cg: cg + dims] = -1
                cg += dims
            return self._jac


class IpoptWrapper:
    """Just the ipopt style stuff"""
    def __init__(self, prob):
        self.prob = prob

    def objective(self, x):
        return self.prob.get_cost(x)

    def gradient(self, x):
        return self.prob.get_gradient(x)

    def constraints(self, x):
        return self.prob.get_constraint(x)

    def jacobian(self, x):
        jac = self.prob.get_jacobian(x, False)
        return jac

    def jacobianstructure(self):
        x = np.zeros(self.prob.dimx)
        return self.prob.get_jacobian(x, True)

class DirectTranscriptionControllerFactory(ControllerFactory):
    """
    Direct Transcription (DT) is a method to discretize an optimal control problem which is inherently continuous.
    Such discretization is usually necessary in order to get an optimization problem of finite dimensionality.
    For a trajectory with time length :math:`T`, it discretize the time interval into a equidistant grid of size :math:`N`, called knots.
    The state and control at each knot are optimized.
    The constraints are imposed at the knots, including system dynamics constraints.
    DT uses first-order Euler integration to approximate the constraints of system dynamics.
    The details can be found in `An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation <https://epubs.siam.org/doi/pdf/10.1137/16M1062569>`_.

    Hyperparameter:
    - *horizon* (Type: int, Lower: 1, High: 30, Default: 10): Control Horizon
    """
    def __init__(self, *args, **kwargs):
        try:
            import cyipopt
        except:
            raise ImportError("Missing dependency for Direct Transcription Controller")
        super().__init__(*args, **kwargs)
        self.Controller = DirectTranscriptionController
        self.name = "DirectTranscription"

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        horizon = CSH.UniformIntegerHyperparameter(name="horizon",
                lower=1, upper=30, default_value=10)
        cs.add_hyperparameter(horizon)
        return cs

class DirectTranscriptionController(Controller):
    """
    Implementation of the linear controller. For this very basic version, it accepts some linear models and compute output.
    constraints is a dict of constraints we have to consider, it has two keys: path and terminal. The items are list of Constraints.
    cost is a Cost instance to compute fitness of a trajectory
    """
    def __init__(self, system, model, task, horizon):
        try:
            import cyipopt
        except:
            raise ImportError("Missing dependency for Direct Transcription Controller")
        Controller.__init__(self, system, model, task)
        self.horizon = int(np.ceil(horizon / system.dt))
        self._built = False
        self._guess = None
        self._x_dim = (self.horizon + 1) * system.obs_dim + self.horizon * system.ctrl_dim

    def reset(self):
        self._built = False
        self._guess = None

    def set_guess(self, guess):
        if guess.size != self._xdim:
            raise Exception("Guess dimension should be %d" % self._x_dim)
        self._guess = guess

    def _build_problem(self):
        """Use cvxpy to construct the problem"""
        self._built = True
        self.problem = NonLinearMPCProblem(self.system, self.model, self.task, self.horizon)
        self.wrapper = IpoptWrapper(self.problem)

    def _update_problem_and_solve(self, x0):
        """Solve the problem"""
        if not self._built:
            self._build_problem()

        dims = self.problem.obs_dim
        lb, ub = self.problem.get_variable_bounds()
        cl, cu = self.problem.get_constr_bounds()
        lb[:dims] = ub[:dims] = x0
        ipopt_prob = cyipopt.Problem(
            n=self.problem.dimx,
            m=self.problem.dimc,
            problem_obj = self.wrapper,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )
        if self._guess is None:
            guess = np.zeros(self.problem.dimx)
        else:
            guess = self._guess

        ipopt_prob.add_option("max_iter", 10)
        sol, info = ipopt_prob.solve(guess)
        return sol, info

    @property
    def state_dim(self):
        return self.model.state_dim + self.model.ctrl_dim

    @staticmethod
    def is_compatible(system, task, model):
        return True  # this should be universal...
 
    def traj_to_state(self, traj):
        return np.concatenate([self.model.traj_to_state(traj),
                traj[-1].ctrl])

    def run(self, state, new_obs):
        x = self.model.update_state(state[:-self.system.ctrl_dim],
                state[-self.system.ctrl_dim:], new_obs)
        self._x_cache = x
        sol, info = self._update_problem_and_solve(x)

        # update guess
        self._guess = sol.copy()
        dims = self.problem.obs_dim
        dimu = self.problem.ctrl_dim
        idx0 = dims * (self.horizon + 1)
        u = sol[idx0: idx0 + dimu]
        statenew = np.concatenate([x, u])

        return u, statenew
