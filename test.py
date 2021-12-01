# import autompc as ampc
# import numpy as np

import autompc as ampc

from autompc.benchmarks import CartpoleSwingupBenchmark

benchmark = CartpoleSwingupBenchmark()

system = benchmark.system
#trainTrajs = benchmark.gen_trajs(seed=100, n_trajs=2, traj_len=4)
trainTrajs = benchmark.gen_trajs(seed=100, n_trajs=200, traj_len=200)
testTrajs = benchmark.gen_trajs(seed=200, n_trajs=30, traj_len=300)


from autompc.sysid import MLPDAD
modeldad = MLPDAD(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=50, #100 is problematic
               nonlintype="relu", n_dad_iters=7, test_trajs=testTrajs)

modeldad.train(trainTrajs)


from autompc.sysid import MLP

model = MLP(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=25,
               nonlintype="relu")

model.train(trainTrajs)

import matplotlib.pyplot as plt
from autompc.graphs.kstep_graph import KstepPredAccGraph

graph = KstepPredAccGraph(system, testTrajs, kmax=40, metric="rmse")
graph.add_model(modeldad, "DaD MLP")
graph.add_model(model, "MLP")

fig = plt.figure()
ax = fig.gca()
graph(fig, ax)
ax.set_title("Comparison of MLP models")
plt.show()
plt.savefig('Best MLP DaD Prediction Error.png', dpi=600, bbox_inches='tight')

# from autompc.benchmarks import CartpoleSwingupBenchmark

# benchmark = CartpoleSwingupBenchmark()

# system = benchmark.system
# trajs = benchmark.gen_trajs(seed=100, n_trajs=5, traj_len=10)

# from autompc.sysid.mlp import transform_input

# print("Traj 0\n" + str(trajs[0].obs))
# X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
# print("X\n" + str(X))
# dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs]) # 
# print("dY\n" + str(dY))
# print("\n")

# for traj in trajs:
#     print(traj.obs[1:,:])
#     print(traj.obs[:-1,:])
#     print(traj.obs[1:,:] - traj.obs[:-1,:])
# U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
# print("U\n" + str(U))
# XU = np.concatenate((X, U), axis = 1) # stack X and U together
# print("XU\n" + str(XU))
# xu_means = np.mean(XU, axis=0)
# xu_std = np.std(XU, axis=0)
# XUt = transform_input(xu_means, xu_std, XU)
# print("XUt" + str(XUt))

# dy_means = np.mean(dY, axis=0)
# dy_std = np.std(dY, axis=0)
# dYt = transform_input(dy_means, dy_std, dY)
# print("dYt" + str(dYt))
# # concatenate data
# feedX = XUt
# predY = dYt

# from autompc.sysid import MLP
# from autompc.sysid import MLPDAD

# # model = MLP(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=3,
# #                nonlintype="relu")

# # model.train(trajs)

# modeldad = MLPDAD(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=10,
#               nonlintype="relu", n_dad_iters=3)

# modeldad.train(trajs)