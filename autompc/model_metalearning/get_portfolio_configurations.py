import os
import pickle
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from ..tuning import ModelTuner
from meta_utils import gym_names, metaworld_names
from meta_utils import load_data
from task_pool import MPITaskPool
from ..sysid import MLP

def run_model_tuning(system, trajs, n_iters=1000):
    # model tuner
    # default evaluatiuon strategy: 3-fold-cv and one-step RMSE
    tuner = ModelTuner(system, trajs, verbose=1, multi_fidelity=False)
    print("Selecting from models", ",".join(model.name for model in tuner.model.models))
    tuned_model, tune_result = tuner.run(n_iters=n_iters, eval_timeout=1000)
    print("Selected model:", tuned_model.name)
    print("Selected configuration", tune_result.inc_cfg)
    print("Final cross-validated RMSE score", tune_result.inc_costs[-1])
    return tuned_model, tune_result

def get_configurations(names):
    PATH = '/home/baoyul2/autompc/autompc/model_metalearning/meta_data'
    for name in names:
        # Get data
        print(name)
        print('Start loading data.')
        system, trajs = load_data(path=PATH, name=name)
        print('Finish loading data.')
        
        # Model tuning 
        start = time.time()
        tuned_model, tune_result = run_model_tuning(system, trajs, verbose=True)
        end = time.time()
        print("Model tuning time", end-start)
        
        # Save the configuration
        cfg_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_cfg'
        data_name = name + '.pkl'
        output_file_name = os.path.join(cfg_path, data_name)
        print("Dumping to ", output_file_name)
        with open(output_file_name, 'wb') as fh:
            pickle.dump(tune_result.inc_cfg, fh)
        
        # Plot train curve
        plt.plot(tune_result.inc_costs)
        plt.title(name)
        plt.ylabel('score')
        plt.xlabel('iteration')
        plt.savefig(name + '_inc_costs' + '_100MLP')
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
    # names = ["HalfCheetah-v2", "Ant-v2", 'assembly-v2']
    # names = [['assembly-v2']]
    # mpi_task_pool = MPITaskPool()
    # mpi_task_pool.run(tasks=names, func=get_configurations)
    names = ["Humanoid-v2"]
    get_configurations(names=names)
    