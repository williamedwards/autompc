# Standard library includes
import unittest
import tempfile
import os
from datetime import datetime

# External library includes
import numpy as np

# Internal library includes
import autompc as ampc
from autompc.tuning import ControlTuner, ControlTunerResult
from autompc.benchmarks import CartpoleSwingupBenchmark
from autompc.sysid import MLP
from autompc import Controller, AutoSelectController
from autompc.sysid import MLP
from autompc.optim import IterativeLQR
from autompc.ocp import QuadCostTransformer

def main():
    benchmark = CartpoleSwingupBenchmark()
    benchmark.task.set_num_steps(200)

    # Generate data
    trajs = benchmark.gen_trajs(seed=0, n_trajs=200, traj_len=200)

    # Set-Up AutoMPC output directory
    if os.getenv("AUTOMPC_OUTPUT_DIR"):
        autompc_dir = os.getenv("AUTOMPC_OUTPUT_DIR")
    else:
        autompc_dir = tempfile.mkdtemp()
    print(f"{autompc_dir=}")

    # Surrogate
    surrogate = MLP(benchmark.system)
    surrogate.freeze_hyperparameters()

    controller = Controller(benchmark.system)
    controller.add_optimizer(IterativeLQR(benchmark.system))
    controller.add_model(MLP(benchmark.system, n_train_iters=10))
    controller.add_ocp_transformer(QuadCostTransformer(benchmark.system))

    start_time = datetime.now()

    tuner = ControlTuner(surrogate=surrogate, surrogate_split=0.5, control_tune_bootstraps=20)
    tuned_controller, tune_result = tuner.run(
        controller, 
        benchmark.task,
        trajs,
        n_iters=2,
        rng = np.random.default_rng(0),
        truedyn=benchmark.dynamics,
        output_dir=autompc_dir
    )

    elapsed_time = datetime.now() - start_time

    print(f"Elapsed Wallclock Time: {elapsed_time}")

if __name__ == "__main__":
    main()