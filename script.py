from distutils.log import info
import os, glob
import pickle
from collections import namedtuple

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger

from autompc.benchmarks import CartpoleSwingupV2Benchmark, benchmark
from autompc import AutoSelectController
from autompc.tuning import ControlTuner
from autompc.sysid import MLP

run_dir = '/home/randomgraph/baoyul2/meta/autompc/autompc-output_2022-08-14T18:04:00/run_1660536240245'
with open(os.path.join(run_dir, "tuning_data.pkl"), "rb") as f:
    tuning_data = pickle.load(f)

# print(tuning_data["controller"])

restore_dir = '/home/randomgraph/baoyul2/meta/autompc/autompc-output_2022-08-14T18:04:00/run_1660536240245'
path = os.path.join(restore_dir)
# print(path)

def _get_restore_run_dir(restore_dir):
    run_dirs = glob.glob(os.path.join(restore_dir))
    # print(run_dirs)
    for run_dir in reversed(sorted(run_dirs)):
        if os.path.exists(os.path.join(run_dir, "smac", "run_1", "runhistory.json")):
            return run_dir
    raise FileNotFoundError("No valid restore files found")

restore_run_dir = _get_restore_run_dir(run_dir)
print(restore_run_dir)

# result['surr_info'] = [trial_to_json(info) for info in result['surr_info']]
control_evaluator = tuning_data["control_evaluator"]

# ## data
# benchmark = CartpoleSwingupV2Benchmark()
# system = benchmark.system
# task   = benchmark.task
# # Generate benchmark dataset
# trajs = benchmark.gen_trajs(seed=100, n_trajs=100, traj_len=200)

# controller = AutoSelectController(system)
# controller.set_ocp(benchmark.task.get_ocp())

# trajs = control_evaluator(controller)
# print(trajs)

ControlEvaluationTrial = namedtuple("ControlEvaluationTrial", ["policy","task","dynamics","weight",
    "cost","traj","term_cond","eval_time"])

info = ControlEvaluationTrial()
info.policy
info.task
info.dynamics
info.weight
info.cost
info.traj
info.term_cond
info.eval_time
