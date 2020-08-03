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

def pendulum_dynamics(y,u,g=9.8,m=1,L=1,b=0.1):
    theta, omega = y
    return [omega,((u[0] - b*omega)/(m*L**2)
        - g*np.sin(theta)/L)]

def dt_pendulum_dynamics(y,u,dt,g=9.8,m=1,L=1,b=0.1):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: pendulum_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((2,))
    y[0] -= np.pi
    return sol.y.reshape((2,))

dt = 0.01

umin = -2.0
umax = 2.0

# Generate trajectories for training
num_trajs = 100

@memory.cache
def gen_trajs(dt):
    rng = np.random.default_rng(42)
    trajs = []
    for _ in range(num_trajs):
        y = [-np.pi, 0.0]
        traj = ampc.zeros(pendulum, 400)
        for i in range(400):
            traj[i].obs[:] = y
            u = rng.uniform(umin, umax, 1)
            y = dt_pendulum_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

trajs = gen_trajs(dt)

X = [traj.obs for traj in trajs]
U = [traj.ctrls for traj in trajs]

# Continuous Time SINDy
identity_library = ps.IdentityLibrary()
fourier_library = ps.FourierLibrary(n_frequencies=1)
combined_library = identity_library + fourier_library
ct_model = ps.SINDy(feature_library=combined_library,
        optimizer=ps.STLSQ(threshold=0.5))
ct_model.fit(X, u=U, multiple_trajectories=True, t=dt)
print("Continuous Time SINDy Results")
print("=============================")
print(ct_model.print())

# Discrete Time SINDy
dt_model = ps.SINDy(feature_library=combined_library, discrete_time=True,
        optimizer=ps.STLSQ(threshold=0.0))
dt_model.fit(X, u=U, multiple_trajectories=True)
print("Discrete Time SINDy Results")
print("=============================")
print(dt_model.print())

from autompc.sysid import SINDy
cs = SINDy.get_configuration_space(pendulum)
s = cs.get_default_configuration()
s["trig_basis"] = "true"
s["trig_freq"] = 1
s["poly_basis"] = "false"
#s["poly_degree"] = 2
model = ampc.make_model(pendulum, SINDy, s)
model.train(trajs)
state = np.array([1.0, 2.0])
ctrl = np.array([3.0])
print("model.pred")
print(model.pred(state.copy(), ctrl))
print("model.pred_diff")
print(model.pred_diff(state.copy(), ctrl))
set_trace()