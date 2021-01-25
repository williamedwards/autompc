# Created by William Edwards, (wre2@illinois.edu)

import numpy as np
from pdb import set_trace

from abc import ABC, abstractmethod

class Constraint(ABC):
    def __init__(self, system, dim, equality):
        self.system = system
        self.equality = equality
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @abstractmethod
    def eval(self, state):
        """
        Maps state -> val (n-array)
        For equality constraints, we require val == 0.
        For inequality constraints, we require val <= 0.
        """
        return self._dim

class Constraints:
    def __init__(self, system, constraint_list, equality):
        self.system = system
        self.constraint_list = constraint_list
        self.equality = equality

    @property
    def dim(self): 
        return sum([cons.dim for cons in self.constraint_list])

    def is_diff(self):
        for cons in self.constraint_list:
            if not isinstance(cons, DiffConstraint):
                return False
        return True

    def is_convex(self):
        for cons in self.constraint_list:
            if not isinstance(cons, ConvexConstraint):
                return False
        return True

    def is_affine(self):
        for cons in self.constraint_list:
            if not isinstance(cons, AffineConstraint):
                return False
        return True

    def eval(self, state):
        vals = []
        for cons in self.constraint_list:
            vals.append(cons.eval(state))
        return np.concatenate(vals)

    def eval_diff(self, state):
        if not self.is_diff():
            return ValueError("Constraints are not differentiable.")
        vals, jacs = [], []
        for cons in self.constraint_list:
            val, jac = cons.eval_diff(state)
            vals.append(val)
            jacs.append(jac)
        if len(vals) > 0:
            return np.concatenate(vals), np.concatenate(jacs)
        else:
            return np.array([]), np.array([[]])

    def get_matrix(self):
        if not self.is_affine():
            return ValueError("Constraints are not affine.")
        return np.concatenate([cons.matrix for cons in self.constraint_list])

class EqConstraints(Constraints):
    def __init__(self, system, constraint_list):
        super().__init__(system, constraint_list, True)

class IneqConstraints(Constraints):
    def __init__(self, system, constraint_list):
        super().__init__(system, constraint_list, False)

class NumericConstraint(Constraint):
    def __init__(self, system, func, dim, equality):
        super().__init__(system, dim, equality)
        self.func = func

    def eval(self, state):
        val = self.func(state)
        if val.shape != (self.dim,):
            raise ValueError("Constraint returned wrong shape")
        return val

class NumericEqConstraint(NumericConstraint):
    def __init__(self, system, func, dim):
        super().__init__(system, func, dim, True)

class NumericIneqConstraint(NumericConstraint):
    def __init__(self, system, func, dim):
        super().__init__(system, func, dim, False)

class DiffConstraint(Constraint):
    def __init__(self, system, func, dim, equality):
        self.func = func
        super().__init__(system, dim, equality)

    def eval(self, state):
        val, _ = self.func(state)
        if val.shape != (self.dim,):
            raise ValueError("Constraint returned wrong shape")
        return val

    def eval_diff(self, state):
        val, jac = self.func(state)
        if val.shape != (self.dim,):
            raise ValueError("Constraint returned wrong shape")
        if jac.shape != (self.dim,self.system.obs_dim):
            raise ValueError("Constraint jacobian is wrong shape.")
        return val, jac

class DiffEqConstraint(DiffConstraint):
    def __init__(self, system, func, dim):
        super().__init__(system, func, dim, True)

class DiffIneqConstraint(DiffConstraint):
    def __init__(self, system, func, dim):
        super().__init__(system, func, dim, False)

class ConvexConstraint(DiffConstraint):
    def __init__(self, system, func, dim, equality):
        super().__init__(system, func, dim, equality)

class ConvexEqConstraint(ConvexConstraint):
    def __init__(self, system, func, dim):
        super().__init__(system, func, dim, True)

class ConvexIneqConstraint(ConvexConstraint):
    def __init__(self, system, func, dim):
        super().__init__(system, func, dim, False)

class AffineConstraint(ConvexConstraint):
    def __init__(self, system, A, equality):
        A = np.copy(A)
        self.matrix = A
        dim = A.shape[0]
        if A.shape != (dim, system.obs_dim):
            raise ValueError("Constraint matrix is wrong shape")
        func = lambda state: (A @ state, np.copy(A))
        super().__init__(system, func, dim, equality)

class AffineEqConstraint(AffineConstraint):
    def __init__(self, system, A):
        super().__init__(system, A, True)

class AffineIneqConstraint(AffineConstraint):
    def __init__(self, system, A):
        super().__init__(system, A, False)


