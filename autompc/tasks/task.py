# Created by William Edwards (wre2@illinois.edu)

import numpy as np

from .cost import BaseCost
from .constraints import EqConstraints, IneqConstraints

class Task:
    def __init__(self, system):
        self.system = system

        # Initialize obs and control bounds
        self._obs_bounds = np.zeros((system.obs_dim, 2))
        for i in range(system.obs_dim):
            self._obs_bounds[i, 0] = -np.inf
            self._obs_bounds[i, 1] = np.inf

        self._ctrl_bounds = np.zeros((system.ctrl_dim, 2))
        for i in range(system.ctrl_dim):
            self._ctrl_bounds[i, 0] = -np.inf
            self._ctrl_bounds[i, 1] = np.inf

        self._eq_cons = []
        self._ineq_cons = []
        self._term_eq_cons = []
        self._term_ineq_cons = []
        self._init_eq_cons = []
        self._init_ineq_cons = []
        
        self._init_obs = None
        self._term_cond = None
        self._num_steps = None

    def set_num_steps(self, num_steps):
        self._term_cond = lambda traj: len(traj) >= num_steps
        self._num_steps = num_steps

    def has_num_steps(self):
        return self._num_steps is not None

    def get_num_steps(self):
        return self._num_steps

    def term_cond(self, traj):
        if self._term_cond is not None:
            return self._term_cond(traj)
        else:
            return False

    def set_term_cond(self, term_cond):
        self._term_cond = term_cond

    def set_cost(self, cost):
        self.cost = cost

    def get_cost(self):
        return self.cost

    def set_init_obs(self, init_obs):
        self._init_obs = init_obs[:]

    def get_init_obs(self):
        if self._init_obs is not None:
            return self._init_obs[:]
        else:
            return None

    # Handle bounds
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

    def are_obs_bounded(self):
        for i in range(self.system.obs_dim):
            if (self._obs_bounds[i, 0] != -np.inf 
                    or self._obs_bounds[i, 1] != np.inf):
                return True
        return False

    def are_ctrl_bounded(self):
        for i in range(self.system.ctrl_dim):
            if (self._ctrl_bounds[i, 0] != -np.inf 
                    or self._ctrl_bounds[i, 1] != np.inf):
                return True
        return False

    def get_obs_bounds(self):
        return self._obs_bounds.copy()

    def get_ctrl_bounds(self):
        return self._ctrl_bounds.copy()

    # Handle Constraints
    def add_eq_constraint(self, cons):
        self._eq_cons.append(cons)

    def add_ineq_constraint(self, cons):
        self._ineq_cons.append(cons)

    def add_term_eq_constraint(self, cons):
        self._term_eq_cons.append(cons)

    def add_term_ineq_constraint(self, cons):
        self._term_ineq_cons.append(cons)

    def add_init_eq_constraint(self, cons):
        self._init_eq_cons.append(cons)

    def add_init_ineq_constraint(self, cons):
        self._init_ineq_cons.append(cons)

    def eq_constraints_present(self):
        return bool(self._eq_cons)

    def ineq_constraints_present(self):
        return bool(self._ineq_cons)

    def term_eq_constraints_present(self):
        return bool(self._term_eq_cons)

    def term_ineq_constraints_present(self):
        return bool(self._term_ineq_cons)

    def init_eq_constraints_present(self):
        return bool(self._init_eq_cons)

    def init_ineq_constraints_present(self):
        return bool(self._init_ineq_cons)

    def get_eq_constraints(self):
        return EqConstraints(self.system, self._eq_cons[:])

    def get_ineq_constraints(self):
        return IneqConstraints(self.system, self._ineq_cons[:])

    def get_term_eq_constraints(self):
        return EqConstraints(self.system, self._term_eq_cons[:])

    def get_term_ineq_constraints(self):
        return IneqConstraints(self.system, self._term_ineq_cons[:])

    def get_init_eq_constraints(self):
        return EqConstraints(self.system, self._init_eq_cons[:])

    def get_init_ineq_constraints(self):
        return IneqConstraints(self.system, self._init_ineq_cons[:])

