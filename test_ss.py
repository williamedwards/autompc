import autompc as ampc
from autompc.benchmarks import CartpoleSwingupBenchmark

benchmark = CartpoleSwingupBenchmark()

system = benchmark.system
trajs = benchmark.gen_trajs(seed=100, n_trajs=200, traj_len=500)

#Params for testing
iters = 1000
batch = 64
samplings = 1000

from autompc.sysid import MLPSS

testTraj = benchmark.gen_trajs(seed=300, n_trajs=1, traj_len=1000)

model = MLPSS(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=iters, n_batch=batch,
               nonlintype="relu", n_sampling_iters=samplings, test_trajectory=testTraj)

model.train(trajs)

#n_hidden_layers=4, hidden_size_1=256, hidden_size_2=256, hidden_size_3=256, hidden_size_4=256,
#n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128,

from autompc.sysid import MLP


model2 = MLP(system, n_hidden_layers=2, hidden_size_1=128, hidden_size_2=128, n_train_iters=iters, n_batch=batch,
               nonlintype="relu")

model2.train(trajs)

import matplotlib.pyplot as plt
from autompc.graphs.kstep_graph import KstepPredAccGraph

    
trajs2 = benchmark.gen_trajs(seed=200, n_trajs=10, traj_len=300)

graph = KstepPredAccGraph(system, trajs2, kmax=100, metric="rmse")
graph.add_model(model, "MLP with SS")
graph.add_model(model2, "MLP")

fig = plt.figure()
ax = fig.gca()
graph(fig, ax)
ax.set_title("Comparison of MLP and MLP SS models: Iters 1000 Batch 64 Samplings 1000 Layers 2 All 128 Trajs 200 by 500")
plt.show()
plt.savefig('MLP SS Iters 1000 Batch 64 Samplings 1000 Layers 2 All 128 Trajs 200 by 500.png', dpi=600, bbox_inches='tight')