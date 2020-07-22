"""
Test the nmpc.py file.
"""
import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt

from autompc.sysid.dummy_nonlinear import DummyNonlinear
from autompc.control import NonLinearMPC
from autompc import Task


dummy_sys = ampc.System(["x1", "x2"], ["u"])
dummy_model = DummyNonlinear(dummy_sys)

task1 = Task(dummy_sys)
Q = np.eye(2)
R = np.eye(1)
F = 10 * np.eye(2)
task1.set_quad_cost(Q, R, F)

horizon = 8
nmpc = NonLinearMPC(dummy_sys, dummy_model, task1, horizon)
# just give a random initial state
sim_traj = ampc.zeros(dummy_sys, 1)
x = np.array([2, 1.0])
sim_traj[0].obs[:] = x

for _ in range(10):
    u, _ = nmpc.run(sim_traj)
    print('u = ', u)
    x = dummy_model.traj_to_state(sim_traj)
    x = dummy_model.pred(x, u)
    sim_traj[-1, "u"] = u
    sim_traj = ampc.extend(sim_traj, [x], [[0.0]])

print(sim_traj[:, "x1"], sim_traj[:, 'x2'])
fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.plot(sim_traj[:, 'x1'], sim_traj[:, 'x2'])
plt.show()
