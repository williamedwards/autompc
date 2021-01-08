import autompc as ampc
from autompc.tasks.task import Task
from autompc.tasks.quad_cost import QuadCost
from autompc.tasks.constraints import *

pendulum = ampc.System(["ang", "angvel"], ["torque"])

task = Task(pendulum)

Q = np.eye(2)
R = np.eye(1)
cost = QuadCost(pendulum, Q, R)
assert(cost.is_quad)
assert(cost.is_convex)
assert(cost.is_diff)

task.set_cost(cost)


def func(state):
    val = np.array([state[0]**2])
    jac = np.array([[2*state[0], 0.0]])
    return val, jac

diff1 = DiffEqConstraint(pendulum, func, 1)
conv1 = ConvexEqConstraint(pendulum, func, 1)
num1 = NumericEqConstraint(pendulum, lambda state: func(state)[0], 1)
A = np.array([[2, 1]])
aff1 = AffineEqConstraint(pendulum, A)
A2 = np.array([[1, 2]])
aff2 = AffineEqConstraint(pendulum, A)

task.add_eq_constraint(aff1)
task.add_eq_constraint(aff2)

cons1 = task.get_eq_constraints()
assert(cons1.is_affine())
assert(cons1.is_convex())
assert(cons1.is_diff())

task.add_term_eq_constraint(conv1)
task.add_term_eq_constraint(aff1)
cons2 = task.get_term_eq_constraints()
assert(not cons2.is_affine())
assert(cons2.is_convex())
assert(cons2.is_diff())






print("All checks passed!")
