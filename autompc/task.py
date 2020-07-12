# Created by William Edwards (wre2@illinois.edu)

import numpy as np
from pdb import set_trace

class Task:
    def __init__(self, system):
        self.system = system

        # Cost types:
        # 0 : None
        # 1 : Quadratic
        # 2 : Convex
        # 3 : Diff
        # 4 : Numeric

        self.cost_type = 0

        # Initialize obs and control bounds
        self._obs_bounds = np.zeros((system.obs_dim, 2))
        for i in range(system.obs_dim):
            self._obs_bounds[i, 0] = -np.inf
            self._obs_bounds[i, 1] = np.inf

        self._ctrl_bounds = np.zeros((system.ctrl_dim, 2))
        for i in range(system.ctrl_dim):
            self._ctrl_bounds[i, 0] = -np.inf
            self._ctrl_bounds[i, 1] = np.inf

        self._affine_eq_cons = []
        self._affine_ineq_cons = []
        self._conv_eq_cons = []
        self._conv_ineq_cons = []
        self._diff_eq_cons = []
        self._diff_ineq_cons = []
        self._eq_cons_dim = 0
        self._ineq_cons_dim = 0

    # Adding Costs
    def set_quad_cost(self, Q, R, F=None):
        if Q.shape != (self.system.obs_dim, self.system.obs_dim):
            raise ValueError("Q is the wrong shape")
        if R.shape != (self.system.ctrl_dim, self.system.ctrl_dim):
            raise ValueError("R is the wrong shape")
        if not F is None:
            if F.shape != (self.system.obs_dim, self.system.obs_dim):
                raise ValueError("F is the wrong shape")
        else:
            F = np.zeros((self.system.obs_dim, self.system.obs_dim))
        self.cost_type = 1
        self._quad_cost = Q, R, F

    def set_convex_cost(self, add_obs_cost, add_ctrl_cost, terminal_obs_cost):
        """
        all arguments are functions with the
        signature
            obs/ctrl -> cost, grad
        """
        self.cost_type = 2
        self._add_obs_cost = add_obs_cost
        self._add_ctrl_cost = add_ctrl_cost
        self._terminal_obs_cost = terminal_obs_cost

    def set_diff_cost(self, add_obs_cost, add_ctrl_cost, terminal_obs_cost):
        """
        all arguments are functions with the
        signature
            obs/ctrl -> cost, grad
        """
        self.cost_type = 3
        self._add_obs_cost = add_obs_cost
        self._add_ctrl_cost = add_ctrl_cost
        self._terminal_obs_cost = terminal_obs_cost

    def set_numeric_cost(self, add_obs_cost, add_ctrl_cost, terminal_obs_cost):
        """
        all arguments are functions with the
        signature
            obs/ctrl -> cost
        """
        self.cost_type = 4
        self._add_obs_cost = add_obs_cost
        self._add_ctrl_cost = add_ctrl_cost
        self._terminal_obs_cost = terminal_obs_cost

    # Cost properties
    def is_cost_quad(self):
        return self.cost_type == 1

    def is_cost_convex(self):
        return self.cost_type <= 2

    def is_cost_diff(self):
        return self.cost_type <= 3

    # Evaluating Costs
    def get_quad_cost(self):
        if self.cost_type != 1:
            raise ValueError("Cost is not quadratic")
        else:
            return self._quad_cost

    def get_costs(self):
        """
        Returns tuple of the following functions:
            add_obs_cost
            add_ctrl_cost
            terminal_obs_cost
        all with the signature
            obs/ctrl -> cost
        """
        if self.cost_type == 1:
            Q, R, F = self._quad_cost
            add_obs_cost = lambda obs: obs.T @ Q @ obs
            add_ctrl_cost = lambda ctrl: ctrl.T @ R @ ctrl
            terminal_obs_cost = lambda ctrl: ctrl.T @ F @ ctrl
        elif self.cost_type == 2 or self.cost_type == 3:
            add_obs_cost = lambda obs: self._add_obs_cost(obs)[0]
            add_ctrl_cost = lambda ctrl: self._add_ctrl_cost(ctrl)[0]
            terminal_obs_cost = lambda obs: self._terminal_obs_cost(obs)[0]
        elif self.cost_type == 4:
            add_obs_cost = self._add_obs_cost
            add_ctrl_cost = self._add_ctrl_cost
            terminal_obs_cost = self._terminal_obs_cost
        elif self.cost_type == 0:
            raise ValueError("Task cost not set")
        return add_obs_cost, add_ctrl_cost, terminal_obs_cost

    def get_costs_diff(self):
        """
        Returns tuple of the following functions:
            add_obs_cost
            add_ctrl_cost
            terminal_obs_cost
        all with the signature
            obs/ctrl -> cost, grad
        """
        if self.cost_type == 1:
            Q, R, F = self._quad_cost
            add_obs_cost = lambda obs: (obs.T @ Q @ obs, (Q + Q.T) @ obs)
            add_ctrl_cost = lambda ctrl: (ctrl.T @ R @ ctrl, (R + R.T) @ ctrl)
            terminal_obs_cost = lambda obs: (obs.T @ F @ obs, (F + F.T) @ obs)
        elif self.cost_type == 2 or self.cost_type == 3:
            add_obs_cost = self._add_obs_cost
            add_ctrl_cost = self._add_ctrl_cost
            terminal_obs_cost = self._terminal_obs_cost
        elif self.cost_type == 4:
            raise ValueError("Cost function is not differentiable")
        elif self.cost_type == 0:
            raise ValueError("Task cost not set")
        return add_obs_cost, add_ctrl_cost, terminal_obs_cost

    # Adding Constraints
    def set_obs_bound(self, obs_label, lower, upper):
        idx = self.system.observations.index(obs_label)
        self._obs_bounds[idx,:] = [lower, upper]

    def set_obs_bounds(self, lowers, uppers):
        self._obs_bounds[:,0] = lowers
        self._obs_bounds[:,1] = uppers

    def set_ctrl_bound(self, ctrl_label, lower, upper):
        idx = self.system.controls.index(ctrl_label)
        self._ctrl_bounds[idx,:] = [lower, upper]

    def set_ctrl_bounds(self, lowers, uppers):
        self._ctrl_bounds[:,0] = lowers
        self._ctrl_bounds[:,1] = uppers

    def add_affine_eq_cons(self, A, b):
        """
        Adds constraint of the form A x = b where
        x is a system observation.
        """
        if A.shape[1] != self.system.obs_dim or A.shape[0] != b.shape[0]:
            raise ValueError("A and b are wrong shape.")
        self._affine_eq_cons.append([A.copy(), b.copy()])
        self._eq_cons_dim += A.shape[0]

    def add_affine_ineq_cons(self, A, b):
        """
        Adds constraint of the form A x <= b where
        x is the model obs.
        """
        if A.shape[1] != self.system.obs_dim or A.shape[0] != b.shape[0]:
            raise ValueError("A and b are wrong shape.")
        self._affine_ineq_cons.append([A.copy(), b.copy()])
        self._ineq_cons_dim += A.shape[0]

    def add_convex_eq_cons(self, func, dim=1):
        """
        func : obs -> np.array of shape (dim,), 
                        np.array of shape(dim, self.system.obs_dim)
        Constraint requires func(obs) = 0
        """
        self._conv_eq_cons.append((func, dim))
        self._eq_cons_dim += dim

    def add_convex_ineq_cons(self, func, dim):
        """
        func : obs -> np.array of shape (dim,), 
                        np.array of shape(dim, self.system.obs_dim)
        Constraint requires func(obs) <= 0
        """
        self._conv_ineq_cons.append((func, dim))
        self._ineq_cons_dim += dim

    def add_diff_eq_cons(self, func, dim=1):
        """
        func : obs -> np.array of shape (dim,), 
                        np.array of shape(dim, self.system.obs_dim)
        Constraint requires func(obs) = 0
        """
        self._diff_eq_cons.append((func, dim))
        self._eq_cons_dim += dim

    def add_diff_ineq_cons(self):
        """
        func : obs -> np.array of shape (dim,), 
                        np.array of shape(dim, self.system.obs_dim)
        Constraint requires func(obs) <= 0
        """
        self._diff_ineq_cons.append((func, dim))
        self._ineq_cons_dim += dim

    #def add_numeric_eq_cons(self):
    #    pass

    #def add_numeric_ineq_cons(self):
    #    pass

    #def add_bool_cons(self):
    #    pass

    # Constraint properties
    def eq_cons_present(self):
        return (self._affine_eq_cons or self._convx_eq_cons
                or self._diff_eq_cons)

    def are_eq_cons_affine(self):
        return (not self._conv_eq_cons and not self._diff_eq_cons)

    def ineq_cons_present(self):
        return (self._affine_ineq_cons or self._conv_eq_cons
                or self._diff_ineq_cons)

    def are_ineq_cons_affine(self):
        return (not self._conv_ineq_cons and not self._diff_eq_cons)

    def are_eq_cons_convex(self):
        return not self._diff_eq_cons

    def are_ineq_cons_convex(self):
        return not self._diff_ineq_cons

    @property
    def eq_cons_dim(self):
        return self._eq_cons_dim

    @property
    def ineq_cons_dim(self):
        return self._ineq_cons_dim


    # Evaluating Constraints
    def get_obs_bounds(self):
        return self._obs_bounds.copy()

    def get_ctrl_bounds(self):
        return self._ctrl_bounds.copy()

    def get_affine_eq_cons(self):
        if not self.are_eq_cons_affine:
            raise ValueError("Equality constraints are not affine.")
        A = np.vstack([A for A, _ in self._affine_eq_cons])
        b = np.concatenate([b for _, b in self._affine_eq_cons])
        return A, b

    def get_affine_ineq_cons(self):
        if not self.are_ineq_cons_affine():
            raise ValueError("Inequality constraints are not affine.")
        A = np.vstack([A for A, _ in self._affine_ineq_cons])
        b = np.concatenate([b for _, b in self._affine_ineq_cons])
        return A, b

    def eval_convex_eq_cons(self, obs):
        if not self.are_eq_cons_convex:
            raise ValueError("Equality constraints are not convex.")
        value = np.zeros((self.eq_cons_dim,))
        grad = np.zeros((self.eq_cons_dim, self.system.obs_dim))
        i = 0
        for A, b in self._affine_eq_cons:
            value[i:i+A.shape[0]] = A @ obs - b
            grad[i:i+A.shape[0], :] = A
            i += A.shape[0]
        for func, dim in self._conv_eq_cons:
            val, gr = func(obs)
            value[i:i+dim] = val
            grad[i:i+dim, :] = gr
            i += dim
        return value, grad

    def eval_diff_eq_cons(self, obs):
        value = np.zeros((self.eq_cons_dim,))
        grad = np.veros((self.eq_cons_dim, self.eq_cons_dim))
        i = 0
        for A, b in self._affine_eq_cons:
            value[i:A.shape[0]] = A @ obs - b
            grad[i:A.shape[0], :] = A
            i += A.shape[0]
        for func, dim in self._conv_eq_cons:
            val, gr = func(obs)
            value[i:dim] = val
            grad[i:dim, :] = gr
            i += dim
        for func, dim in self._diff_eq_cons:
            val, gr = func(obs)
            value[i:dim] = val
            grad[i:dim, :] = gr
            i += dim
        return value, grad

    def eval_convex_ineq_cons(self, obs):
        if not self.are_ineq_cons_convex:
            raise ValueError("Equality constraints are not convex.")
        value = np.zeros((self.ineq_cons_dim,))
        grad = np.zeros((self.ineq_cons_dim, self.system.obs_dim))
        i = 0
        for A, b in self._affine_ineq_cons:
            value[i:i+A.shape[0]] = A @ obs - b
            grad[i:i+A.shape[0], :] = A
            i += A.shape[0]
        for func, dim in self._conv_ineq_cons:
            val, gr = func(obs)
            value[i:i+dim] = val
            grad[i:i+dim, :] = gr
            i += dim
        return value, grad

    def eval_diff_ineq_cons(self, obs):
        value = np.zeros((self.ineq_cons_dim,))
        grad = np.veros((self.ineq_cons_dim, self.ineq_cons_dim))
        i = 0
        for A, b in self._affine_ineq_cons:
            value[i:A.shape[0]] = A @ obs - b
            grad[i:A.shape[0], :] = A
            i += A.shape[0]
        for func, dim in self._conv_ineq_cons:
            val, gr = func(obs)
            value[i:dim] = val
            grad[i:dim, :] = gr
            i += dim
        for func, dim in self._diff_ineq_cons:
            val, gr = func(obs)
            value[i:dim] = val
            grad[i:dim, :] = gr
            i += dim
        return value, grad


