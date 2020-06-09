import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
from pdb import set_trace

pendulum = ampc.System(["ang", "angvel"], ["torque"])

task1 = ampc.Task(pendulum)
Q = np.eye(2)
R = np.eye(1)
task1.set_quad_cost(Q, R)
print(task1.get_quad_cost())

add_state_cost, add_ctrl_cost, terminal_state_cost = task1.get_costs_diff()

print(add_state_cost(np.array([1.0, 2.0])))
print(terminal_state_cost(np.array([1.0, 2.0])))
print(add_ctrl_cost(np.array([3.0])))

