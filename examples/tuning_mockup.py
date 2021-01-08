# Created by William Edwards



pipeline = FixedControlPipeline(cartpole, task, Koopman, FiniteHorizonLQR, 
    QuadCostTransformer])

from autompc.control_evaluation import CrossDataEvaluator
from autompc.control_evaluation import FixedInitialMetric

init_states = [np.array([0.0, 0.0, 0.0, 0.0]),
               np.array([0.2, 0.0, 0.0, 0.0]),
               np.array([0.4, 0.0, 0.0, 0.0]),
               np.array([0.6, 0.0, 0.0, 0.0]),
               np.array([0.8, 0.0, 0.0, 0.0]),
               np.array([1.0, 0.0, 0.0, 0.0]),
               np.array([1.2, 0.0, 0.0, 0.0]),
               np.array([1.4, 0.0, 0.0, 0.0]),
               np.array([1.6, 0.0, 0.0, 0.0]),
               np.array([1.8, 0.0, 0.0, 0.0]),
               np.array([2.0, 0.0, 0.0, 0.0])]

metric = FixedInitialMetric(cartpole, task, init_states, sim_iters)

training_trajs = trajs[:100]
validation_trajs = trajs[200:]

from autompc.evaluators import HoldoutEvaluator
from autompc.metrics import RmseKstepMetric

evaluator = CrossDataEvaluator(cartpole, task, metric, 
        HoldoutEvaluator, {"holdout_prop" : 0.25},
        tuning_iters=30, rng, training_trajs, validation_trajs)

eval_cfg = evaluator(pipeline)

print(eval_cfg(cfg1))
print(eval_cfg(cfg2))
print(eval_cfg(cfg3))
print(eval_cfg(cfg4))
print(eval_cfg(cfg5))
