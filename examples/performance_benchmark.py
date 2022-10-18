# Standard library includes
import unittest
import tempfile
import os
import timeit
from datetime import datetime, timedelta

# External library includes
import numpy as np
import torch

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

    # Train model
    # model = MLP(system=benchmark.system, n_train_iters=4)
    model = MLP(system=benchmark.system)

    start_time = datetime.now()
    model.train(trajs)
    model_train_elapsed = datetime.now() - start_time
    
    print("Model Train Time: ", model_train_elapsed)

    # Model Inference
    BATCH_SIZE = 100
    states = torch.rand(BATCH_SIZE, benchmark.system.obs_dim, dtype=torch.double)
    controls = torch.rand(BATCH_SIZE, benchmark.system.ctrl_dim, dtype=torch.double)

    timeit_number = 10000
    model_inference_elapsed = timeit.timeit(lambda: model.pred_batch(states, controls), number=timeit_number)
    model_inference_individual = timedelta(seconds=model_inference_elapsed/timeit_number)

    print("Model Inference Time: ", model_inference_individual)

    # Build controller
    controller = Controller(benchmark.system)

    controller.set_model(model)
    controller.set_optimizer(IterativeLQR(benchmark.system))
    controller.set_ocp_transformer(QuadCostTransformer(benchmark.system))
    controller.set_ocp(benchmark.task)

    controller.build()

    # Simulate controller
    start_time = datetime.now()
    sim_traj = benchmark.task.simulate(controller, benchmark.dynamics)
    simulate_elapsed = datetime.now() - start_time

    print("Simulation Time: ", simulate_elapsed)

    # Summary
    print("")
    print("=========================================")
    print("Model Train Time: ", model_train_elapsed)
    print("Model Inference Time: ", model_inference_individual)
    print("Simulation Time: ", simulate_elapsed)

if __name__ == "__main__":
    main()