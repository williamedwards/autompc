from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

simplesys = ampc.System(["x1", "x2"], ["u"])

A = np.array([[1, -1], [0, 1]])
B = np.array([[0], [1]])

def sim_simplesys(x0, us):
    traj = ampc.zeros(simplesys, len(us) + 1)
    traj[0].obs[:] = x0

    for i, u in enumerate(us):
        traj[i, "u"] = u
        traj[i+1].obs[:] = A @ traj[i].obs + B @ [u]

    return traj

rng = np.random.default_rng(42)
samples = 100
length = 30

trajs = []
for _ in range(samples):
    x0 = rng.uniform(-10, 10, 2)
    us = rng.uniform(-1, 1, length)
    traj = sim_simplesys(x0, us)
    trajs.append(traj)

traj = trajs[-1]


print(traj.obs)

print(traj.ctrls)

from autompc.sysid import ARX, Koopman

koop = Koopman(simplesys)
koop.train(trajs)


# Test prediction

predobs, _ = koop.pred(traj[:10])
assert(np.allclose(predobs, traj[10].obs))
print("completed assert")

koop_A, koop_B, state_func, cost_func = koop.to_linear()

state = state_func(traj[:10])

state = koop_A @ state + koop_B @ traj[9].ctrl
state = koop_A @ state + koop_B @ traj[10].ctrl

assert(np.allclose(state, traj[11].obs))

Q, R = np.eye(2), np.eye(1)
Qt, Rt, = cost_func(Q, R)
print(Qt, Rt)

from autompc.control import InfiniteHorizonLQR

Q = np.eye(2)
R = np.eye(1)
con = InfiniteHorizonLQR(simplesys, koop, Q, R)

sim_traj = ampc.zeros(simplesys, 1)
x = np.array([1,1])
sim_traj[0].obs[:] = x

for _ in range(30):
    u, _ = con.run(sim_traj)
    x = A @ x + B @ u
    sim_traj[-1, "u"] = u
    sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
    xtrans = con.state_func(sim_traj)

plt.plot(sim_traj[:,"x1"], sim_traj[:,"x2"], "b-o")
plt.show()
