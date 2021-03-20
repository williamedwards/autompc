# Created by William Edwards

# Standard library includes
import sys
import pickle
from pdb import set_trace

# External project includes
import numpy as np
#
#sys.path.append("/home/william/proj/microsurgery_data/src")
#from autoregression import AutoregModel

# Internal project includes
import autompc as ampc
from autompc.evaluators import SimpleEvaluator
from autompc.metrics import RmseKstepMetric
from autompc.sysid import ARX



def runexp_sysid1(Model, tinf, tune_iters, seed):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    training_set = sysid_trajs[:int(0.7*len(sysid_trajs))]
    validation_set = sysid_trajs[int(0.7*len(sysid_trajs)):int(0.85*len(sysid_trajs))]
    testing_set = sysid_trajs[int(0.85*len(sysid_trajs)):]
    
    tuning_evaluator = SimpleEvaluator(tinf.system, training_set, validation_set,
            horiz=int(1/tinf.system.dt))
    final_evaluator = SimpleEvaluator(tinf.system, training_set, testing_set,
            horiz=int(1/tinf.system.dt))

    #model_path = "/home/william/proj/microsurgery_data/experiment_scripts/model2.pkl"
    #with open(model_path, "rb") as f:
    #    ar_model = pickle.load(f)
    #set_trace()
    # cs = Model.get_configuration_space(tinf.system)
    # cfg = cs.get_default_configuration()
    # ar_score = tuning_evaluator(Model, cfg)
    # print("ar_score = ", ar_score)

    tuner = ampc.ModelTuner(tinf.system, tuning_evaluator)
    tuner.add_model(Model)
    tune_result = tuner.run(rng=np.random.RandomState(rng.integers(1 << 30)),
            runcount_limit=tune_iters, n_jobs=1)
    test_score = final_evaluator(Model, tune_result["inc_cfg"])[0] 

    return test_score, tune_result
