from unicodedata import name
import numpy as np

from autompc.benchmarks import CartpoleSwingupV2Benchmark, benchmark
from autompc import AutoSelectController
from autompc.tuning import ControlTuner
from autompc.sysid import MLP

def run_main(benchmark):
    # Get system and task specification
    system = benchmark.system
    task   = benchmark.task
    # Generate benchmark dataset
    trajs = benchmark.gen_trajs(seed=100, n_trajs=100, traj_len=200)
    
    controller = AutoSelectController(system)
    controller.set_ocp(benchmark.task.get_ocp())
    # print(controller.get_config_space())

    # Surrogate model
    tuner = ControlTuner(surrogate=MLP(system), surrogate_split=0.5, surrogate_tune_horizon=5)

    restore_dir = '/home/randomgraph/baoyul2/meta/autompc/autompc-output_2022-08-14T18:04:00/run_1660536240245'
    # End-to-end tuning
    # tuned_controller, tune_result = tuner.run(controller, task, trajs, n_iters=100, rng=np.random.default_rng(100), 
    #                                 truedyn=benchmark.dynamics, restore_dir=restore_dir)
    # tuned_controller, tune_result = tuner.run(controller, task, trajs, n_iters=2, surrogate_tune_iters=2, 
    #                                 rng=np.random.default_rng(100),
    #                                 restore_dir=restore_dir)
    tuned_controller, tune_result = tuner.run(controller, task, trajs, n_iters=2, surrogate_tune_iters=2, 
                                    rng=np.random.default_rng(100))
                                    

    return tuned_controller, tune_result

if __name__ == "__main__":
    benchmark = CartpoleSwingupV2Benchmark()
    tuned_controller, tune_result = run_main(benchmark)
    print(tune_result.inc_cfg)
