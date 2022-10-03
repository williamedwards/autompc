from cProfile import label
import imp
from re import M
import autompc as ampc
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from autompc.benchmarks import benchmark
from autompc.benchmarks.halfcheetah import HalfcheetahBenchmark
from autompc.tuning import ModelTuner
from autompc.benchmarks.meta_benchmarks.reacher import ReacherBenchmark
from autompc.benchmarks.meta_benchmarks.idp import IDPBenchmark
from autompc.benchmarks.meta_benchmarks.ant import AntBenchmark
from autompc.benchmarks.meta_benchmarks.humanoid import HumanoidBenchmark
from autompc.benchmarks.meta_benchmarks.humanoidStandup import HumanoidStandupBenchmark
from autompc.benchmarks import CartpoleSwingupBenchmark
from autompc.benchmarks.meta_benchmarks.metaworld import MetaBenchmark

def run_model_tuning(benchmark):
    # get system specification and generate benchmark dataset
    system = benchmark.system
    trajs = benchmark.gen_trajs(seed=100, n_trajs=100, traj_len=200)

    # model tuner
    # default evaluatiuon strategy: 3-fold-cv and one-step RMSE
    tuner = ModelTuner(system, trajs, verbose=1)
    print("Selecting from models", ",".join(model.name for model in tuner.model.models))
    tuned_model, tune_result = tuner.run(n_iters=100)

    print("Selected model:", tuned_model.name)
    print("Selected configuration", tune_result.inc_cfg)
    print("Final cross-validated RMSE score", tune_result.inc_costs[-1])
    return tuned_model, tune_result

if __name__ == "__main__":
    # names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2",
    #           "Reacher-v2", "InvertedPendulum-v2", "InvertedDoublePendulum-v2", 
    #           "Ant-v2", "Humanoid-v2", "HumanoidStandup-v2"]
    # benchmarks = [IDPBenchmark()]
    # for benchmark in benchmarks:

    # Meta-World
    # names = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 
    #             'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 
    #             'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2']
    # for name in names:
    #     # name = benchmark.name
    #     benchmark = MetaBenchmark(name=name)
        # start = time.time()
        # tuned_model, tune_result = run_model_tuning(benchmark)
        # end = time.time()
        # print("time", end-start)
        
        # # Plot train curve
        # plt.plot(tune_result.inc_costs)
        # plt.title(name)
        # plt.ylabel('score')
        # plt.xlabel('iteration')
        # plt.savefig(name + '_inc_costs')
        # plt.close()

        # # Save information
        # info = {
        #     'env': name,
        #     'time': end-start,
        #     'final_score': tune_result.inc_costs[-1],
        #     'final_config': dict(tune_result.inc_cfg),
        #     'costs': tune_result.costs,
        #     'inc_costs': tune_result.inc_costs
        # }
        # with open('meta_world.json', 'a') as outfile:
        #     outfile.write(json.dumps(info, indent=2))
    
        benchmark = benchmark = CartpoleSwingupBenchmark()
        name = 'cart_pole'
        print(name)
        start = time.time()
        tuned_model, tune_result = run_model_tuning(benchmark)
        end = time.time()
        print("time", end-start)
        
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
            'final_config': dict(tune_result.inc_cfg),
            'costs': tune_result.costs,
            'inc_costs': tune_result.inc_costs
        }
        with open('meta_world.json', 'a') as outfile:
            outfile.write(json.dumps(info, indent=2))

    
    