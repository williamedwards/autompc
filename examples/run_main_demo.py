import autompc as ampc
import numpy as np
import matplotlib.pyplot as plt

from autompc.benchmarks import CartpoleSwingupV2Benchmark
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
    print(controller.get_config_space())
    tuner = ControlTuner(surrogate=MLP(system), surrogate_split=0.5, surrogate_tune_horizon=5)

    tuned_controller, tune_result = tuner.run(controller, task, trajs, n_iters=1, surrogate_tune_iters=1,
                                    rng=np.random.default_rng(100), 
                                   truedyn=benchmark.dynamics)
    
    return tuned_controller, tune_result


if __name__ == "__main__":
    benchmark = CartpoleSwingupV2Benchmark()
    tuned_controller, tune_result = run_main(benchmark)
    print(tune_result.inc_cfg)