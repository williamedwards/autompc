"""
Test the nmpc.py file.
"""
import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt

from autompc.control.mpc import LQRCost
from autompc.sysid.dummy_nonlinear import DummyNonlinear
from autompc.control import NonLinearMPC


dummy_sys = ampc.System(["x1", "x2"], ["u"])
dummy_model = DummyNonlinear(dummy_sys)
cost = LQRCost(Q=np.eye(2), R=np.eye(1), F=10*np.eye(2))
nmpc = NonLinearMPC(dummy_model, dummy_model, cost, constraints=None)
# just give a random initial state
sim_traj = ampc.zeros(dummy_sys, 1)
x = np.array([1, 0.0])
sim_traj[0].obs[:] = x

for _ in range(10):
    u, _ = nmpc.run(sim_traj)
    x = dummy_model.traj_to_state(sim_traj)
    x = dummy_model.pred(x, u)
    sim_traj[-1, "u"] = u
    sim_traj = ampc.extend(sim_traj, [x], [[0.0]])

fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.plot(sim_traj[:, 'x1'], sim_traj[:, 'x2'])
plt.show()
