import os
import pickle
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from autompc.tuning import ModelTuner

# from autompc.benchmarks import benchmark
# from autompc.benchmarks.halfcheetah import HalfcheetahBenchmark
# from autompc.benchmarks.meta_benchmarks.reacher import ReacherBenchmark
# from autompc.benchmarks.meta_benchmarks.idp import IDPBenchmark
# from autompc.benchmarks.meta_benchmarks.ant import AntBenchmark
# from autompc.benchmarks.meta_benchmarks.humanoid import HumanoidBenchmark
# from autompc.benchmarks.meta_benchmarks.humanoidStandup import HumanoidStandupBenchmark
# from autompc.benchmarks import CartpoleSwingupBenchmark
# from autompc.benchmarks.meta_benchmarks.metaworld import MetaBenchmark

def run_model_tuning(system, trajs, n_iters=100):
    # model tuner
    # default evaluatiuon strategy: 3-fold-cv and one-step RMSE
    tuner = ModelTuner(system, trajs, verbose=1, multi_fidelity=False)
    print("Selecting from models", ",".join(model.name for model in tuner.model.models))
    tuned_model, tune_result = tuner.run(n_iters=n_iters)
    print("Selected model:", tuned_model.name)
    print("Selected configuration", tune_result.inc_cfg)
    print("Final cross-validated RMSE score", tune_result.inc_costs[-1])
    return tuned_model, tune_result

if __name__ == "__main__":
    # openai gyms
    gym_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2",
              "Reacher-v2", "InvertedPendulum-v2", "InvertedDoublePendulum-v2", 
              "Ant-v2", "Humanoid-v2", "HumanoidStandup-v2"]

    # Meta-World
    metaworld_names = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 
                'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 
                'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2']
    
    names = ["HalfCheetah-v2", "Ant-v2", 'assembly-v2']
    
    PATH = '/home/baoyul2/autompc/autompc/model_metalearning/meta_data'
    
    for name in names:
        # Get data
        print(name)
        print('Start loading data.')
        data_name = name + '.pkl'
        input_file_name = os.path.join(PATH, data_name)
        with open(input_file_name, 'rb') as fh:
            data = pickle.load(fh)
            system = data['system']
            trajs = data['trajs']
        print('Finish loading data.')
        
        start = time.time()
        tuned_model, tune_result = run_model_tuning(system, trajs)
        end = time.time()
        print("Model tuning time", end-start)
        
        # Plot train curve
        plt.plot(tune_result.inc_costs)
        plt.title(name)
        plt.ylabel('score')
        plt.xlabel('iteration')
        plt.savefig(name + '_inc_costs')
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
    
        

    
    