from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
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

A, B, state_func, cost_func = arx.to_linear()

state = state_func(traj[:10])

state = A @ state + B @ traj[10].ctrl
state = A @ state + B @ traj[11].ctrl

assert(np.allclose(state[-3:-1], traj[11].obs))

Q, R = np.eye(2), np.eye(1)
Qt, Rt, = cost_func(Q, R)
print(Qt, Rt)
