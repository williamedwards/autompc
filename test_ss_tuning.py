import autompc as ampc
import numpy as np
import matplotlib.pyplot as plt

from autompc.benchmarks import CartpoleSwingupV2Benchmark
from autompc.benchmarks import CartpoleSwingupBenchmark

benchmark = CartpoleSwingupV2Benchmark()

system = benchmark.system
task   = benchmark.task

trajs = benchmark.gen_trajs(seed=100, n_trajs=400, traj_len=200)

from autompc.evaluation import HoldoutModelEvaluator

model_evaluator = HoldoutModelEvaluator(holdout_prop=0.25, metric="rmse", horizon=50, trajs=trajs, 
                                        system=system, rng=np.random.default_rng(100))

from autompc.sysid import MLPSSFactory
model_factory = MLPSSFactory(system)

from autompc.tuning import ModelTuner

model_tuner = ModelTuner(system, model_evaluator)
model_tuner.add_model_factory(model_factory)

model, model_tune_result = model_tuner.run(rng=np.random.default_rng(100), n_iters=300)

print(model_tune_result)
#print(str(model.get_configuration_space()))

from autompc.graphs import TuningCurveGraph
import matplotlib.pyplot as plt

graph = TuningCurveGraph()

fig = plt.figure()      
ax = fig.gca()
graph(ax, model_tune_result)
ax.set_title("Model Tuning Curve")
plt.savefig('Model Tuning Curve.png', dpi=600, bbox_inches='tight')
#plt.show()
plt.clf()

import matplotlib.pyplot as plt
from autompc.graphs.kstep_graph import KstepPredAccGraph

    
trajs2 = benchmark.gen_trajs(seed=200, n_trajs=50, traj_len=300)

graph = KstepPredAccGraph(system, trajs2, kmax=50, metric="rmse")
graph.add_model(model, "MLP with SS tuned result")

fig = plt.figure()
ax = fig.gca()
graph(fig, ax)
ax.set_title("Best Tuned Result RMSE")
plt.savefig('Best Tuned Result.png', dpi=600, bbox_inches='tight')
plt.show()