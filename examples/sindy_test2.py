from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Memory

from scipy.integrate import solve_ivp
import pysindy as ps

memory = Memory("cache")

pendulum = ampc.System(["ang", "angvel"], ["torque"])

def dt_dynamics(y,u,dt=0.1):
    x1, x2 = y
    ynew = np.zeros_like(y)
    ynew[0] = x1 + dt*x2
    ynew[1] = x2 - dt*x1*x2 + dt*u

    return ynew

dt = 0.1

umin = -2.0
umax = 2.0

# Generate trajectories for training
num_trajs = 100

@memory.cache
def gen_trajs(dt):
    rng = np.random.default_rng(42)
    trajs = []
    for _ in range(num_trajs):
        y = [0.0, 0.0]
        traj = ampc.zeros(pendulum, 400)
        for i in range(400):
            traj[i].obs[:] = y
            u = rng.uniform(umin, umax, 1)
            y = dt_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
            if abs(y[0]) > 1e6 or abs(y[1]) > 1e6:
                traj = traj[:i+1]
                break
        trajs.append(traj)
    return trajs

trajs = gen_trajs(dt)

X = [traj.obs for traj in trajs]
U = [traj.ctrls for traj in trajs]

from autompc.sysid import SINDy
cs = SINDy.get_configuration_space(pendulum)
s = cs.get_default_configuration()
s["trig_basis"] = "false"
#s["trig_freq"] = 1
s["poly_basis"] = "true"
s["poly_degree"] = 2
s["poly_cross_terms"] = "true"
model = ampc.make_model(pendulum, SINDy, s)
model.train(trajs)
state = np.array([1.0, 2.0])
ctrl = np.array([3.0])
print("model.pred")
print(model.pred(state.copy(), ctrl))
print("model.pred_diff")
print(model.pred_diff(state.copy(), ctrl))
set_trace()
