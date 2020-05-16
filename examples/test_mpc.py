"""
Test autompc framework with mpc controller and arx model.
"""
from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import matplotlib.pyplot as plt
import autompc as ampc

from scipy.integrate import solve_ivp

simplesys = ampc.System(["x1", "x2"], ["u"])

A = np.array([[1, -1], [0, 1]])
B = np.array([[0], [1]])

def sim_simplesys(x0, us):
    traj = ampc.zeros(simplesys, len(us) + 1)
    traj[0].obs[:] = x0

    for i, u in enumerate(us):
        traj[i, "u"] = u
        traj[i + 1].obs[:] = A @ traj[i].obs + B @ [u]

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

from autompc.control import LinearMPC
from autompc.control.mpc import MPCConstraints, LQRCost

from autompc.sysid import ARX, Koopman

arx = ARX(simplesys)

print(arx.is_linear)

print(arx.get_hyper_options())

print(arx.get_hypers())

arx.set_hypers(k=2)
print(arx.get_hypers())

arx.train(trajs)

# Test prediction

predobs, _ = arx.pred(traj[:10])
assert(np.allclose(predobs, traj[10].obs))

arx_A, arx_B, state_func, cost_func = arx.to_linear()

state = state_func(traj[:10])

state = arx_A @ state + arx_B @ traj[10].ctrl
state = arx_A @ state + arx_B @ traj[11].ctrl

assert(np.allclose(state[-3:-1], traj[11].obs))

Q, R = np.eye(2), np.eye(1)
cost = LQRCost(Q, R)
con = LinearMPC(simplesys, arx, cost, None)
print(con.is_diff)
print(con.get_hyper_options())
print(con.get_hypers())

con.set_hypers(horizon=6)
print(con.get_hypers())

# sim_traj = ampc.zeros(simplesys, 1)
# x = np.array([1, 1])
# sim_traj[0].obs[:] = x
# for _ in range(30):
#     u, _ = con.run(sim_traj)
#     x = A @ x + B @ u
#     sim_traj[-1, "u"] = u
#     sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
#     xtrans = con.state_func(sim_traj)
# 
# plt.figure()
# plt.plot(sim_traj[:,"x1"], sim_traj[:,"x2"], "b-o")
# plt.show()

# do a koopman
koop = Koopman(simplesys)
koop.set_hypers()
koop.train(trajs)

Q, R = np.eye(2), np.eye(1)
cost = LQRCost(Q, R)
con = LinearMPC(simplesys, koop, cost, None)
con.set_hypers(horizon=5)

sim_traj = ampc.zeros(simplesys, 1)
x = np.array([1, 1])
sim_traj[0].obs[:] = x

for _ in range(30):
    u, _ = con.run(sim_traj)
    x = A @ x + B @ u
    sim_traj[-1, "u"] = u
    sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
    xtrans = con.state_func(sim_traj)

plt.plot(sim_traj[:,"x1"], sim_traj[:,"x2"], "b-o")
plt.show()
