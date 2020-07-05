# Created by William Edwards (wre2@illinois.edu)

from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt

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


#print(traj.obs)
#
#print(traj.ctrls)

from autompc.sysid import ARX

cs = ARX.get_configuration_space(simplesys)
s = cs.get_default_configuration()
model = ampc.make_model(simplesys, ARX, s)
model.train(trajs)

from autompc.evaluators import HoldoutEvaluator
from autompc.metrics import RmseKstepMetric
from autompc.evaluator import CachingPredictor

metric = RmseKstepMetric(simplesys, k=5)
predictor = CachingPredictor(trajs[-1], model)
score = metric(predictor, trajs[-1])
print("score = {}".format(score))
