from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Memory

from scipy.integrate import solve_ivp

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

def animate_pendulum(fig, ax, dt, traj):
    ax.grid()

    line, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        line.set_data([0.0, np.sin(traj[i,"ang"]+np.pi)], 
                [0.0, -np.cos(traj[i,"ang"] + np.pi)])
        time_text.set_text('t={:2f}'.format(dt*i))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

dt = 0.025

umin = -15.0
umax = 15.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 10

@memory.cache
def gen_trajs(dt, umin, umax, udmax, num_trajs):
    rng = np.random.default_rng(42)
    trajs = []
    for _ in range(num_trajs):
        y = [-np.pi, 0.0]
        traj = ampc.zeros(pendulum, 400)
        u = rng.uniform(umin, umax, 1)
        for i in range(400):
            traj[i].obs[:] = y
            u += rng.uniform(-udmax, udmax, 1)
            if u[0] > umax:
                u[0] = umax
            if u[0] < umin:
                u[0] = umin
            y = dt_pendulum_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs
trajs = gen_trajs(dt, umin, umax, udmax, num_trajs)

from autompc.sysid import ARX, Koopman

Model = Koopman

cs = Model.get_configuration_space(pendulum)
s = cs.get_default_configuration()
model = ampc.make_model(pendulum, Model, s)
model.train(trajs)

from autompc.evaluators import HoldoutEvaluator
from autompc.metrics import RmseKstepMetric
from autompc.graphs import KstepGrapher

metric = RmseKstepMetric(pendulum, k=50)
grapher = KstepGrapher(pendulum, kmax=50, kstep=5, evalstep=10)

rng = np.random.default_rng(42)
evaluator = HoldoutEvaluator(pendulum, trajs, metric, rng, holdout_prop=0.25) 
evaluator.add_grapher(grapher)
eval_score, _, graphs = evaluator(Model, s)
print("eval_score = {}".format(eval_score))
fig = plt.figure()
ax = fig.gca()
graphs[0](ax)
plt.show()
sys.exit(0)

tuner = ampc.ModelTuner(pendulum, evaluator)
tuner.add_model(ARX)
tuner.add_model(Koopman)
ret_value = tuner.run(rng=np.random.RandomState(42), runcount_limit=50,
        n_jobs=10)

print(ret_value)

fig = plt.figure()
ax = fig.gca()
ax.plot(range(len(ret_value["inc_costs"])), ret_value["inc_costs"])
ax.set_title("Incumbent cost over time")
ax.set_ylim([0.0, 5.0])
ax.set_xlabel("Iterations.")
ax.set_ylabel("Cost")
plt.show()

#from smac.scenario.scenario import Scenario
#from smac.facade.smac_hpo_facade import SMAC4HPO
#
#scenario = Scenario({"run_obj": "quality",  
#                     "runcount-limit": 50,  
#                     "cs": cs,  
#                     "deterministic": "true",
#                     "n_jobs" : 10
#                     })
#
#smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
#        tae_runner=lambda cfg: evaluator(Model, cfg)[0])
#
#incumbent = smac.optimize()
#
#print("Done!")
#
#print(incumbent)
