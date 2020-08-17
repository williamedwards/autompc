import numpy as np
import autompc as ampc
from autompc.tasks.quad_cost import QuadCost

pendulum = ampc.System(["ang", "angvel"], ["torque"])

task = ampc.Task(pendulum)

Q = np.eye(2)
R = np.eye(1)

cost = QuadCost(pendulum, Q, R)
assert(cost.is_quad)
assert(cost.is_convex)
assert(cost.is_diff)
Q2, R2, F2 = cost.get_cost_matrices()
assert((Q2 == Q).all())
assert((R2 == R).all())
assert((F2 == np.zeros((2,1))).all())

from autompc.tasks.constraint import *


def func(state):
    val = np.array([state[0]**2])
    jac = np.array([[2*state[0], 0.0]])
    return val, jac

diff1 = DiffEqConstraint(pendulum, func, 1)
conv1 = ConvexEqConstraint(pendulum, func, 1)
num1 = NumericEqConstraint(pendulum, lambda state: func(state)[0], 1)
A = np.array([[2, 1]])
aff1 = AffineEqConstraint(pendulum, A)

cons1 = EqConstraints(pendulum, [aff1, aff1])
assert((cons1.get_matrix() == np.array([[2,1],[2,1]])).all())
assert(cons1.is_affine())
assert(cons1.is_convex())
assert(cons1.is_diff())

cons2 = EqConstraints(pendulum, [conv1, aff1])
assert(not cons2.is_affine())
assert(cons2.is_convex())
assert(cons2.is_diff())
val, jac = cons2.eval_diff([2, 2])
print(val)
assert((val == np.array([4, 6])).all())
assert((jac == np.array([[4,0], [2,1]])).all())

cons3 = EqConstraints(pendulum, [conv1, aff1, diff1])
assert(not cons3.is_convex())

cons4 = EqConstraints(pendulum, [conv1, aff1, diff1, num1])
assert(not cons4.is_diff())
val = cons4.eval([2, 2])
assert((val == np.array([4, 6, 4, 4])).all())

print("All checks passed!")
