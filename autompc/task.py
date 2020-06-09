# Created by William Edwards (wre2@illinois.edu)

class Task:
    def __init__(self, system):
        self.system = system

        # Initialize task properties
        self._valid = False
        self._qp = False
        self._conv = False

        # Cost types:
        # 0 : None
        # 1 : Quadratic
        # 2 : Convex
        # 3 : Diff
        # 4 : Numeric

        self.cost_type = 0

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

    def set_convex_cost(self, add_state_cost, add_ctrl_cost, terminal_state_cost):
        """
        all arguments are functions with the
        signature
            state/ctrl -> cost, grad
        """
        self.cost_type = 2
        self._add_state_cost = add_state_cost
        self._add_ctrl_cost = add_ctrl_cost
        self._terminal_state_cost = terminal_state_cost

    def set_diff_cost(self, add_state_cost, add_ctrl_cost, terminal_state_cost):
        """
        all arguments are functions with the
        signature
            state/ctrl -> cost, grad
        """
        self.cost_type = 3
        self._add_state_cost = add_state_cost
        self._add_ctrl_cost = add_ctrl_cost
        self._terminal_state_cost = terminal_state_cost

    def set_numeric_cost(self, add_state_cost, add_ctrl_cost, terminal_state_cost):
        """
        all arguments are functions with the
        signature
            state/ctrl -> cost
        """
        self.cost_type = 4
        self._add_state_cost = add_state_cost
        self._add_ctrl_cost = add_ctrl_cost
        self._terminal_state_cost = terminal_state_cost

    # Adding Constraints
    def add_state_bound(self, state, lower, upper):
        pass

    def add_control_bound(self):
        pass

    def add_affine_eq_cons(self):
        pass

    def add_affine_ineq_cons(self):
        pass

    def add_convex_eq_cons(self):
        pass

    def add_convex_ineq_cons(self):
        pass

    def add_diff_eq_cons(self):
        pass

    def add_diff_ineq_cons(self):
        pass

    def add_numeric_eq_cons(self):
        pass

    def add_numeric_ineq_cons(self):
        pass

    def add_bool_cons(self):
        pass

    # Evaluating Costs
    def get_quad_cost(self):
        pass

    def eval_cost(self, traj):
        pass

    def eval_cost_diff(self):
        pass

    # Evaluating Constraints
    def get_state_bounds(self):
        pass

    def get_linear_state_bounds(self):
        pass
