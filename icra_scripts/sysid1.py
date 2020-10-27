# Created by William Edwards

# Standard library includes

# External project includes
import numpy as np

# Internal project includes
import autompc as ampc
from autompc.evaluators import FixedSetEvaluator
from autompc.metrics import RmseKstepMetric


def runexp_sysid1(Model, tinf, tune_iters, seed):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    training_set = sysid_trajs[:int(0.7*len(sysid_trajs))]
    validation_set = sysid_trajs[int(0.7*len(sysid_trajs)):int(0.85*len(sysid_trajs))]
    testing_set = sysid_trajs[int(0.85*len(sysid_trajs)):]
    
    metric = RmseKstepMetric(tinf.system, k=int(1/tinf.system.dt))
    tuning_evaluator = FixedSetEvaluator(tinf.system, training_set + validation_set, 
            metric, rng, training_trajs=training_set)
    final_evaluator = FixedSetEvaluator(tinf.system, training_set + testing_set, 
            metric, rng, training_trajs=training_set)

    tuner = ampc.ModelTuner(tinf.system, tuning_evaluator)
    tuner.add_model(Model)
    tune_result = tuner.run(rng=np.random.RandomState(rng.integers(1 << 30)),
            runcount_limit=tune_iters, n_jobs=1)

    test_score = final_evaluator(Model, tune_result["inc_cfg"])[0] 

    return test_score, tune_result
