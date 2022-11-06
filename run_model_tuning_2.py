import os
import pickle
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from autompc.tuning import ModelTuner
from autompc.model_metalearning.meta_utils import gym_names, metaworld_names
from autompc.model_metalearning.meta_utils import load_data
from autompc.sysid import MLP

def run_model_tuning(system, trajs, n_iters=100):
    # model tuner
    # default evaluatiuon strategy: 3-fold-cv and one-step RMSE
    # tuner = ModelTuner(system, trajs, MLP(system), verbose=1, multi_fidelity=False)
    tuner = ModelTuner(system, trajs, verbose=1, multi_fidelity=False)
    print("Selecting from models", ",".join(model.name for model in tuner.model.models))
    tuned_model, tune_result = tuner.run(n_iters=n_iters)
    print("Selected model:", tuned_model.name)
    print("Selected configuration", tune_result.inc_cfg)
    print("Final cross-validated RMSE score", tune_result.inc_costs[-1])
    return tuned_model, tune_result

def get_configurations(names):
    PATH = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/gym_mujoco_data'
    for name in names:
        # Get data
        print(name)
        print('Start loading data.')
        system, trajs = load_data(path=PATH, name=name)
        print('Finish loading data.')
        
        # Model tuning 
        start = time.time()
        tuned_model, tune_result = run_model_tuning(system, trajs)
        end = time.time()
        print("Model tuning time", end-start)
        
        # Plot train curve
        plt.plot(tune_result.inc_costs)
        plt.title(name + '_100')
        plt.ylabel('score')
        plt.xlabel('iteration')
        plt.savefig(name + '_inc_costs' + '_100')
        plt.close()

        # Save information
        info = {
            'env': name,
            'time': end-start,
            'final_score': tune_result.inc_costs[-1],
            'final_config': dict(tune_result.inc_cfg)
        }
        with open('test.json', 'a') as outfile:
            outfile.write(json.dumps(info, indent=2))
    

if __name__ == "__main__":
    # gravity_names = ["Walker2dGravityHalf-v2", "Walker2dGravityThreeQuarters-v2", "Walker2dGravityOneAndQuarter-v2", "Walker2dGravityOneAndHalf-v0"]
    gravity_names = ["Walker2dGravityThreeQuarters-v2"]
    get_configurations(names=gravity_names)
    