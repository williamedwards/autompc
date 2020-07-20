from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Memory

from scipy.integrate import solve_ivp
from scipy import io

memory = Memory("cache")

#pendulum = ampc.System(["ang", "angvel"], ["torque"])
fish_system = ampc.System(["x", "y", "ang", "xvel", "yvel", "angvel"], 
        ["u1", "u2"])

# Load fish trajectories
data_file = "example_data/InterpolatedData_200Hz.mat"

mat = io.loadmat(data_file, squeeze_me=True)
x = mat['x_int_list']
y = mat['y_int_list']
theta = mat['psi_int_list']
xvel = mat['v1_int_list']
yvel = mat['v2_int_list']
omega = mat['omega_int_list']
u1 = mat['u1_list']
u2 = mat['u2_list']

traj_boundaries = [0]
for i in range(1, x.shape[0]):
    if u1[i] != u1[i-1] or u2[i] != u2[i-1]:
        traj_boundaries.append(i)
traj_boundaries.append(x.shape[0])

trajs = []
for traj_start, traj_end in zip(traj_boundaries[:-1], traj_boundaries[1:]):
    traj = ampc.zeros(fish_system, traj_end - traj_start)
    traj.obs[:] = np.c_[x,y,theta,xvel,yvel,omega][traj_start:traj_end]
    traj.ctrls[:] = np.c_[u1, u2][traj_start:traj_end]
    trajs.append(traj)

from autompc.sysid import ARX, Koopman

Model = ARX

cs = Model.get_configuration_space(fish_system)
s = cs.get_default_configuration()
#s["method"] = "lstsq"
#s["poly_basis"] = "true"
#s["poly_degree"] = 4
#s["trig_basis"] = "false"
s["history"] = 3
model = ampc.make_model(fish_system, Model, s)
model.train(trajs)


def plot_traj_rollout(ax, model, traj, start, obsvar):
    state = model.traj_to_state(traj[:start+1])
    obs_idx = model.system.observations.index(obsvar)
    predobs = np.zeros(len(traj)-start)
    for i in range(len(traj)-start):
        predobs[i] = state[obs_idx]
        state = model.pred(state, traj[start+i].ctrl)
    actualobs = traj[start:][:, obsvar]
    ax.plot(list(range(start, len(traj))), actualobs, "g-")
    ax.plot(list(range(start, len(traj))), predobs, "r-")

fig = plt.figure()
ax = fig.gca()
plot_traj_rollout(ax, model, trajs[10], 10, "x")
plt.show()

set_trace()

from autompc.evaluators import HoldoutEvaluator
from autompc.metrics import RmseKstepMetric

metric = RmseKstepMetric(fish_system, k=1000, step=25)

rng = np.random.default_rng(42)
evaluator = HoldoutEvaluator(fish_system, trajs, metric, rng, holdout_prop=0.25) 
eval_score = evaluator(Model, s)
print("eval_score = {}".format(eval_score))

tuner = ampc.ModelTuner(fish_system, evaluator)
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
