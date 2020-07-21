import numpy as np
from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")
import autompc as ampc

pendulum = ampc.System(["ang", "angvel"], ["torque"])

task = ampc.Task(pendulum)

Q = np.eye(2)
R = np.eye(1)

# Set costs
task.set_quad_cost(Q, R)
assert(task.is_cost_quad())
assert(task.is_cost_convex())
assert(task.is_cost_diff())

# Add state and control bounds
task.set_obs_bound("angvel", -1.0, 1.0)
task.set_ctrl_bound("torque", -0.5, 0.5)

assert((task.get_obs_bounds() == np.array([[-np.inf, np.inf], [-1.0, 1.0]])).all())
assert((task.get_ctrl_bounds() == np.array([[-0.5, 0.5]])).all())

# Add equality constraints
task.add_affine_eq_cons(np.array([[1, -1]]), np.array([1]))
task.add_affine_eq_cons(np.array([[0, 0]]), np.array([0]))

assert((task.get_affine_eq_cons()[0] == np.array([[1, -1], [0, 0]])).all())
assert((task.get_affine_eq_cons()[1] == np.array([1, 0])).all())
assert(task.are_eq_cons_affine())
assert(task.are_eq_cons_convex())
assert((task.eval_convex_eq_cons(np.array([6, 3]))[0] == np.array([2, 0])).all())
assert((task.eval_convex_eq_cons(np.array([6, 3]))[1] 
    == np.array([[1, -1], [0, 0]])).all())

assert(not task.initial_obs_fixed())
assert(not task.terminal_obs_fixed())

task.fix_initial_obs(np.array([1, 0]))

assert(task.initial_obs_fixed())
assert((task.get_fixed_initial_obs() == np.array([1,0])).all())
assert((task.get_initial_affine_eq_cons()[0] == np.array([[1, -1], [0, 0],
    [1, 0], [0, 1]])).all())
assert((task.get_initial_affine_eq_cons()[1] == np.array([1, 0, 1, 0])).all())

def convex_cons(obs):
    value = np.array([obs[0]**2 - obs[1]**2])
    grad = np.array([2 * obs[0], -2 * obs[1]])
    return value, grad
task.add_convex_eq_cons(convex_cons, dim=1)

assert(task.eq_cons_dim == 3)
assert(task.initial_eq_cons_dim == 5)
assert(not task.are_eq_cons_affine())
assert(not task.are_initial_eq_cons_affine())
assert(task.are_eq_cons_convex())
assert((task.eval_convex_eq_cons(np.array([6, 3]))[0] == np.array([2, 0, 27])).all())
assert((task.eval_initial_convex_eq_cons(np.array([6, 3]))[0] == 
    np.array([2, 0, 5, 3, 27])).all())
assert((task.eval_initial_diff_eq_cons(np.array([6, 3]))[0] == 
    np.array([2, 0, 5, 3, 27])).all())
assert((task.eval_convex_eq_cons(np.array([6, 3]))[1] 
    == np.array([[1, -1], [0, 0], [12, -6]])).all())
assert((task.eval_initial_convex_eq_cons(np.array([6, 3]))[1] 
    == np.array([[1, -1], [0, 0], [1, 0], [0, 1], [12, -6]])).all())

# Add inequality constraints
task.add_affine_ineq_cons(np.array([[1, -1]]), np.array([1]))
task.add_affine_ineq_cons(np.array([[0, 0]]), np.array([0]))

assert((task.get_affine_ineq_cons()[0] == np.array([[1, -1], [0, 0]])).all())
assert((task.get_affine_ineq_cons()[1] == np.array([1, 0])).all())
assert(task.are_ineq_cons_affine())
assert(task.are_ineq_cons_convex())
assert((task.eval_convex_ineq_cons(np.array([6, 3]))[0] == np.array([2, 0])).all())
assert((task.eval_convex_ineq_cons(np.array([6, 3]))[1] 
    == np.array([[1, -1], [0, 0]])).all())

task.add_initial_affine_ineq_cons(np.array([[3, 2]]), np.array([2]))
assert((task.get_affine_ineq_cons()[0] == np.array([[1, -1], [0, 0]])).all())
assert((task.get_affine_ineq_cons()[1] == np.array([1, 0])).all())
assert((task.get_initial_affine_ineq_cons()[0] == np.array([[1, -1], [0, 0], [3,2]]))
        .all())
assert((task.get_initial_affine_ineq_cons()[1] == np.array([1, 0, 2])).all())

def convex_cons2(obs):
    value = np.array([2*obs[0]**2 - obs[1]**2])
    grad = np.array([4 * obs[0], -2 * obs[1]])
    return value, grad
task.add_convex_ineq_cons(convex_cons, dim=1)
task.add_terminal_convex_ineq_cons(convex_cons2, dim=1)

assert(task.ineq_cons_dim == 3)
assert(not task.are_ineq_cons_affine())
assert(task.are_ineq_cons_convex())
assert((task.eval_convex_ineq_cons(np.array([6, 3]))[0] == np.array([2, 0, 27])).all())
assert((task.eval_terminal_convex_ineq_cons(np.array([6, 3]))[0] 
    == np.array([2, 0, 27, 63])).all())
assert((task.eval_convex_ineq_cons(np.array([6, 3]))[1] 
    == np.array([[1, -1], [0, 0], [12, -6]])).all())
assert((task.eval_terminal_convex_ineq_cons(np.array([6, 3]))[1] 
    == np.array([[1, -1], [0, 0], [12, -6], [24, -6]])).all())



print("All checks passed!")
