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

def dynamics(y,u):
    x1, x2 = y
    return np.array([x2, -x1*x2 + u[0]])

#def dt_dynamics(y,u,dt=0.1):
#    dy = dynamics(y,u)
#    return y + dt*dy

def dt_dynamics(y,u,dt=0.1):
    sol = solve_ivp(lambda t, y: dynamics(y,u),
            (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.flatten()
    return y

#def dt_dynamics(y,u,dt=0.1):
#    x1, x2 = y
#    ynew = np.zeros_like(y)
#    ynew[0] = x1 + dt*x2
#    ynew[1] = x2 - dt*x1*x2 + dt*u
#
#    return ynew

dt = 0.01
pendulum.dt = dt

umin = -2.0
umax = 2.0

# Generate trajectories for training
num_trajs = 100

@memory.cache
def gen_trajs(seed, dt):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajs):
        y = [0.0, 0.0]
        traj = ampc.zeros(pendulum, 400)
        for i in range(400):
            traj[i].obs[:] = y
            u = rng.uniform(umin, umax, 1)
            y = dt_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
            if abs(y[0]) > 10 or abs(y[1]) > 10:
                traj = traj[:i+1]
                break
        trajs.append(traj)
    return trajs

trajs = gen_trajs(43, dt)

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
s["time_mode"] = "continuous"
model = ampc.make_model(pendulum, SINDy, s)
xdot = [np.array([dynamics(traj[i].obs,traj[i].ctrl) for i in range(len(traj))])
                for traj in trajs]
model.train(trajs, xdot=xdot)
state = np.array([1.0, 2.0])
ctrl = np.array([3.0])
print("model.pred")
print(model.pred(state.copy(), ctrl))
print("model.pred_diff")
print(model.pred_diff(state.copy(), ctrl))
set_trace()
