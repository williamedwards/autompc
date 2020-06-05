# Created by William Edwards (wre2@illinois.edu)

class Task:
    def __init__(self, system):
        self.system = system

        # Initialize task properties
        self._valid = False
        self._qp = False
        self._conv = False

    # Adding Costs
    def set_quad_cost(self):
        pass

    def set_convex_cost(self):
        pass

    def set_diff_cost(self):
        pass

    def set_numeric_cost(self):
        pass

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
