import os
os.chdir('/home/baoyu/baoyul2/autompc')
import autompc as ampc
import numpy as np
import pickle
import os
import json

from autompc.benchmarks.meta_benchmarks.gym_mujoco import GymExtensionBenchmark

name = "HumanoidStandupAndRunWithSensor-v2"
# name = "HalfCheetahGravityHalf-v2"
benchmark = GymExtensionBenchmark(name=name)
system = benchmark.system
trajs = benchmark.gen_trajs(seed=100, n_trajs=100, traj_len=200)

X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
XU = np.concatenate((X, U), axis = 1) # stack X and U together
print("X {}; U {}; XU {}".format(X.shape, U.shape, XU.shape))

def transform_input(xu_means, xu_std, XU):
    XUt = []
    for i in range(XU.shape[1]):
        if xu_std[i] == 0:
            print(i)
            
            XUt.append(XU[:,i] - xu_means[i])
        else:
            XUt.append((XU[:,i] - xu_means[i]) / xu_std[i])
    return np.vstack(XUt).T

xu_means = np.mean(XU, axis=0)
xu_std = np.std(XU, axis=0)
print(len(xu_std))
print(xu_std)

# Save information
info = {
    'env': name,
    'data': list(xu_std)
}
with open('data.json', 'a') as outfile:
    outfile.write(json.dumps(info, indent=2))