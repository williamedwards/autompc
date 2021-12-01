import autompc as ampc
from autompc.benchmarks import CartpoleSwingupBenchmark

benchmark = CartpoleSwingupBenchmark()

system = benchmark.system
trajs = benchmark.gen_trajs(seed=100, n_trajs=200, traj_len=200)



from autompc.sysid import MLPSS

testTraj = benchmark.gen_trajs(seed=300, n_trajs=1, traj_len=1000)

model = MLPSS(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=1000,
               nonlintype="relu", n_sampling_iters=5, test_trajectory=testTraj)

model.train(trajs)




from autompc.sysid import MLP


model2 = MLP(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=1000,
               nonlintype="relu")

model2.train(trajs)

import matplotlib.pyplot as plt
from autompc.graphs.kstep_graph import KstepPredAccGraph


trajs2 = benchmark.gen_trajs(seed=200, n_trajs=50, traj_len=300)

graph = KstepPredAccGraph(system, trajs2, kmax=50, metric="rmse")
graph.add_model(model, "MLP with SS")
graph.add_model(model2, "MLP")

fig = plt.figure()
ax = fig.gca()
graph(fig, ax)
ax.set_title("Comparison of MLP models")
plt.show()