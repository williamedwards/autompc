from unicodedata import name
import numpy as np

from autompc.benchmarks import CartpoleSwingupBenchmark
from autompc.tuning import ModelTuner

def run_model_tuning(benchmark):
    # Get system specification
    system = benchmark.system
    # Generate benchmark dataset
    trajs = benchmark.gen_trajs(seed=100, n_trajs=100, traj_len=200)

